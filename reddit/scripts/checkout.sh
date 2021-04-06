if [ ! -d ./deps ]
then
	mkdir ./deps
fi

pushd ./deps
git clone https://github.com/ProsusAI/finBERT.git
mkdir -p finBERT/models/classifier_model/finbert-sentiment
git lfs install
git clone https://huggingface.co/ProsusAI/finbert finBERT/models/classifier_model/finbert-sentiment
popd
