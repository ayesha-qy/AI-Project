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
