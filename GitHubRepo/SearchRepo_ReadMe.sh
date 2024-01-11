#!/bin/bash

#GITHUB_TOKEN=
#EXPRESSIONS=("OneClassSVM")
GITHUB_TOKEN=$1
EXPRESSIONS=($2)

rest_call() {
    curl --request GET \
        --url "https://api.github.com/search/code?q=${EXPRESSION}+language:python&page=${page}" \
        --header "Accept: application/vnd.github.v3+json" \
        --header "Authorization: Bearer ${GITHUB_TOKEN}"
}     

getHeader (){
    curl -I -s --request GET \
    --url "https://api.github.com/search/code?q="${EXPRESSION}"" \
    --header "Accept: application/vnd.github.v3+json" \
    --header "Authorization: Bearer ${GITHUB_TOKEN}"
}
 
for i in ${!EXPRESSIONS[@]};
do
    EXPRESSION=${EXPRESSIONS[i]}
    if [ -z "$EXPRESSION" ]
    then
      echo "[Expression not set]" 
      exit 0
    fi

    getHeader > header.txt && sleep 60
    echo "[SLEEPING AFTER HEADER]"
    headerFile=header.txt

    last_page=$(grep '^link:' header.txt | sed -e 's/^link:.*page=//g' -e 's/>.*$//g')
    page=1
    if [[ -f $headerFile ]]; then
        code=$(grep "^HTTP" header.txt)
        if [[ "$code" == *"200"* ]]; 
        then
            [ -z "$last_page" ] && { echo "[LAST PAGE NOT SET]"; exit 1;} 
            echo "[Header Call Successful]"
            while [ "$page" -lt "$last_page" ]
            do 
                echo "CURRENT ITERATION ${page}, STOPS AT ${last_page}"
                echo "[Making Call... Waiting for 30 seconds]" && sleep 30
                rest_call > response.json && response=response.json
                if grep "secondary rate limit" "$response"; 
                then
                    echo "SECONDARY RATE LIMITED, WAITING:" && sleep 30
                    echo "Retrying..."
                    continue
                fi
                jq ".items[].repository.html_url?" response.json >> links.txt
                rm ./response.json
                page=$((page+1))
            done
        elif [[ "$code" == *"403"* ]]; then
            echo "[ERROR] 403 Code"
            exit 0
        elif [[ "$code" == *"401"* ]]; then
            echo "[ERROR] 401 Code"
            exit 0
        elif [[ "$code" == *"422"* ]]; then
            echo "[ERROR] 422 Code"
            exit 0
        else 
            echo "[ERROR] OTHER "
            rm ./header.txt
            exit 0
        fi
    else
        echo "[ERROR] header does not exist"
        exit 0
    fi
    echo "[Successfully Completed]"
    echo "[Cleaning Directory]"
    uniq links.txt >> output.txt
    rm ./links.txt
done    

rm ./header.txt
rm ./links.txt