#!/bin/bash

# GITHUB_TOKEN=""
GITHUB_TOKEN=$1

# CSV file to store the results
CSV_FILE="repository_stars.csv"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GitHub personal access token is missing."
    exit 1
fi

# Header for the CSV file
echo "Repository URL,Star Count" > "$CSV_FILE"

# Read the list of repository URLs from the file
while IFS= read -r REPO_URL; do
    # Skip empty lines
    [ -z "$REPO_URL" ] && continue

    # Remove leading and trailing whitespaces and any extra double quotes
    REPO_URL=$(echo "$REPO_URL" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')

    # Extract the username and repository name from the URL using awk and remove extra double quotes
    USERNAME=$(echo "$REPO_URL" | awk -F'/' '{print $4}')
    REPOSITORY=$(echo "$REPO_URL" | awk -F'/' '{print $5}' | sed 's/"//g')

    # Make the API request to get the repository information with the personal access token
    RESPONSE=$(curl --write-out "%{http_code}" --silent --output /dev/null --request GET \
        --url "https://api.github.com/repos/${USERNAME}/${REPOSITORY}" \
        --header "Accept: application/vnd.github.v3+json" \
        --header "Authorization: Bearer ${GITHUB_TOKEN}")

    if [ "$RESPONSE" -eq 200 ]; then
        # API request successful, extract the star count using jq
        STAR_COUNT=$(curl --silent --request GET \
            --url "https://api.github.com/repos/${USERNAME}/${REPOSITORY}" \
            --header "Accept: application/vnd.github.v3+json" \
            --header "Authorization: Bearer ${GITHUB_TOKEN}" | jq -r '.stargazers_count')

        if [ "$STAR_COUNT" == "null" ]; then
            echo "Failed to retrieve star count for ${USERNAME}/${REPOSITORY}"
        else
            # Output the result to the CSV file
            echo "${REPO_URL},${STAR_COUNT}" >> "$CSV_FILE"
            echo "Star count for ${REPO_URL}: ${STAR_COUNT}"
        fi
    elif [ "$RESPONSE" -eq 401 ]; then
        # Unauthorized access
        echo "Unauthorized access to ${USERNAME}/${REPOSITORY}. Please check your GitHub personal access token."
        exit 1
    else
        # API request failed
        echo "Failed to retrieve information for ${USERNAME}/${REPOSITORY}. HTTP Status Code: $RESPONSE"
    fi


done < output.txt

echo "Results saved to $CSV_FILE"

