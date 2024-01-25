# For each file, add a download.py line
# Any additional processing on the downloaded file

$HERE = Get-Location

# Yelp Reviews dataset
mkdir "$HERE\yelp" -Force
if (-not (Test-Path "$HERE\yelp\raw_train.csv")) {
    python download.py "1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM" "$HERE\yelp\raw_train.csv" # 12536
}
if (-not (Test-Path "$HERE\yelp\raw_test.csv")) {
    python download.py "1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB" "$HERE\yelp\raw_test.csv" # 4
}
if (-not (Test-Path "$HERE\yelp\reviews_with_splits_lite.csv")) {
    python download.py "1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB" "$HERE\yelp\reviews_with_splits_lite.csv" # 1217
}

# Surnames Dataset
mkdir "$HERE\surnames" -Force
if (-not (Test-Path "$HERE\surnames\surnames.csv")) {
    python download.py "1MBiOU5UCaGpJw2keXAqOLL8PCJg_uZaU" "$HERE\surnames\surnames.csv" # 6
}
if (-not (Test-Path "$HERE\surnames\surnames_with_splits.csv")) {
    python download.py "1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU-" "$HERE\surnames\surnames_with_splits.csv" # 8
}

# Books Dataset
mkdir "$HERE\books" -Force
if (-not (Test-Path "$HERE\books\frankenstein.txt")) {
    python download.py "1XvNPAjooMyt6vdxknU9VO_ySAFR6LpAP" "$HERE\books\frankenstein.txt" # 14
}
if (-not (Test-Path "$HERE\books\frankenstein_with_splits.csv")) {
    python download.py "1dRi4LQSFZHy40l7ZE85fSDqb3URqh1Om" "$HERE\books\frankenstein_with_splits.csv" # 109
}

# AG News Dataset
mkdir "$HERE\ag_news" -Force
if (-not (Test-Path "$HERE\ag_news\news.csv")) {
    python download.py "1hjAZJJVyez-tjaUSwQyMBMVbW68Kgyzn" "$HERE\ag_news\news.csv" # 188
}
if (-not (Test-Path "$HERE\ag_news\news_with_splits.csv")) {
    python download.py "1Z4fOgvrNhcn6pYlOxrEuxrPNxT-bLh7T" "$HERE\ag_news\news_with_splits.csv" # 208
}

mkdir "$HERE\nmt" -Force
if (-not (Test-Path "$HERE\nmt\eng-fra.txt")) {
    python download.py "1o2ac0EliUod63sYUdpow_Dh-OqS3hF5Z" "$HERE\nmt\eng-fra.txt" # 292
}
if (-not (Test-Path "$HERE\nmt\simplest_eng_fra.csv")) {
    python download.py "1jLx6dZllBQ3LXZkCjZ4VciMQkZUInU10" "$HERE\nmt\simplest_eng_fra.csv" # 30
}
