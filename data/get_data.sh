# check if the folder exists
if [ ! -d "VMMRdb" ]; then

    # download the data
    echo "Downloading data..."
    wget https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=1 -O VMMRdb.zip

    # unzip the data
    echo "Unzipping data..."
    unzip VMMRdb.zip -d VMMRdb

    # remove the zip file
    echo "Removing zip file..."
    rm VMMRdb.zip
else
    echo "Data already exists."
fi
