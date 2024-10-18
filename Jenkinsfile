pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                
                git branch: 'main', url: 'https://github.com/ayesha-qy/AI Project.git'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Run Unit Tests for Product Recommendation') {
            steps {
                
                sh 'python -m unittest discover -s tests -p "test_recommend_products.py"'
            }
        }

        stage('Run Unit Tests for Product Recommendation with Clustering') {
            steps {
                
                sh 'python -m unittest discover -s tests -p "test_recommend_products_with_clustering.py"'
            }
        }

        stage('Deploy Application') {
            steps {
                DEPLOY_DIR="deploy”
                DATA_DIR="data"

                echo "Deploying to $DEPLOY_DIR..."

                if [ ! -d "$DEPLOY_DIR" ]; then
                  mkdir -p "$DEPLOY_DIR"
                fi
                
                cp -r ./src/* "$DEPLOY_DIR/"
                cp -r ./$DATA_DIR "$DEPLOY_DIR/"
                
                source venv/bin/activate
                pip install -r requirements.txt
                
                cd "$DEPLOY_DIR"
                python main.py
                
                echo "Deployment completed successfully!"
            }
        }
    }

    post {
        // Send notifications on build result
        success {
            echo 'Build completed successfully!'
        }
        failure {
            echo 'Build failed!'
        }
    }
}
