provider "aws" {
  region = "us-east-1"
}

##########################
# VPC, Subnets, & IGW
##########################
resource "aws_vpc" "chatbot_vpc" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.chatbot_vpc.id
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.chatbot_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_subnet" "chatbot_subnet_1" {
  vpc_id                  = aws_vpc.chatbot_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "chatbot_subnet_2" {
  vpc_id                  = aws_vpc.chatbot_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "chatbot_subnet_3" {
  vpc_id                  = aws_vpc.chatbot_vpc.id
  cidr_block              = "10.0.3.0/24"
  availability_zone       = "us-east-1c"
  map_public_ip_on_launch = true
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.chatbot_subnet_1.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.chatbot_subnet_2.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "c" {
  subnet_id      = aws_subnet.chatbot_subnet_3.id
  route_table_id = aws_route_table.public_rt.id
}

##########################
# Security Groups
##########################

# Security group for the ALB – allows HTTP from anywhere.
resource "aws_security_group" "chatbot_alb_sg" {
  name        = "chatbot-alb-sg"
  description = "Allow HTTP traffic to the ALB"
  vpc_id      = aws_vpc.chatbot_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for the ECS container – only allow traffic from the ALB.
resource "aws_security_group" "chatbot_sg" {
  name        = "chatbot-sg"
  description = "Allow traffic from the ALB to the chatbot container"
  vpc_id      = aws_vpc.chatbot_vpc.id

  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.chatbot_alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

##########################
# Application Load Balancer (ALB)
##########################
resource "aws_lb" "chatbot_alb" {
  name               = "chatbot-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.chatbot_alb_sg.id]
  subnets            = [
    aws_subnet.chatbot_subnet_1.id,
    aws_subnet.chatbot_subnet_2.id,
    aws_subnet.chatbot_subnet_3.id
  ]
}

resource "aws_lb_target_group" "chatbot_tg" {
  name        = "chatbot-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.chatbot_vpc.id
  target_type = "ip"

  health_check {
    path                = "/"
    protocol            = "HTTP"
    matcher             = "200-399"
    interval            = 60
    timeout             = 5
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "chatbot_listener" {
  load_balancer_arn = aws_lb.chatbot_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.chatbot_tg.arn
  }
}

##########################
# ECS Cluster & IAM Role
##########################
resource "aws_ecs_cluster" "chatbot_cluster" {
  name = "chatbot-cluster"
}

resource "aws_iam_role" "ecsTaskExecutionRole" {
  name = "ecsTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  ]
}

##########################
# ECS Task Definition
##########################
variable "hf_api_token" {
  description = "Hugging Face API token"
  type        = string
  sensitive   = true
}

resource "aws_ecs_task_definition" "chatbot_task" {
  family                   = "chatbot-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "4096"
  memory                   = "16386"
  execution_role_arn       = aws_iam_role.ecsTaskExecutionRole.arn

  container_definitions = jsonencode([{
    name      = "chatbot"
    image     = "axelsirota/rag_chatbot_inference:prod"
    essential = true
    portMappings = [{
      containerPort = 8080,
      hostPort      = 8080
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = "/ecs/chatbot"
        awslogs-region        = "us-east-1"
        awslogs-stream-prefix = "ecs"
      }
    }
    environment = [
      {
        name  = "HF_API_TOKEN"
        value = var.hf_api_token
      }
    ]
  }])
}

##########################
# ECS Service
##########################
resource "aws_ecs_service" "chatbot_service" {
  name            = "chatbot-service"
  cluster         = aws_ecs_cluster.chatbot_cluster.id
  task_definition = aws_ecs_task_definition.chatbot_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [
      aws_subnet.chatbot_subnet_1.id,
      aws_subnet.chatbot_subnet_2.id,
      aws_subnet.chatbot_subnet_3.id
    ]
    security_groups  = [aws_security_group.chatbot_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.chatbot_tg.arn
    container_name   = "chatbot"
    container_port   = 8080
  }
}

##########################
# CloudWatch Log Group
##########################
resource "aws_cloudwatch_log_group" "ecs_log_group" {
  name              = "/ecs/chatbot"
  retention_in_days = 7
}

##########################
# Terraform Output
##########################
output "alb_dns" {
  description = "The ALB DNS name. Use this domain in your browser to access the chatbot UI."
  value       = aws_lb.chatbot_alb.dns_name
}
