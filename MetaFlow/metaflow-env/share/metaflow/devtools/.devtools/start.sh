#!/bin/bash
set -e
trap "exit" INT TERM
trap "kill 0" EXIT
eval $(/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/minikube/minikube docker-env)
if [ -n "$SERVICES_OVERRIDE" ]; then
    echo "🌐 Using user-provided list of services: $SERVICES_OVERRIDE"
    SERVICES="$SERVICES_OVERRIDE"
else
    echo "📝 Selecting services..."
    SERVICES=$(/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/pick_services.sh)
fi
PATH="/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/minikube:/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/tilt:$PATH" /home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/minikube/minikube tunnel &
echo -e "🚀 Starting Tilt with selected services..."
echo -e "\033[1;38;5;46m\n🔥 \033[1;38;5;196mNext Steps:\033[0;38;5;46m Use \033[3mmetaflow-dev shell\033[23m to switch to the development\n   environment's shell and start executing your Metaflow flows.\n\033[0m"
PATH="/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/helm:/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/minikube:/home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools/.devtools/tilt:$PATH" SERVICES="$SERVICES" tilt up -f /home/ozkan/Desktop/personal_projects/personal-projects/MetaFlow/metaflow-env/share/metaflow/devtools//Tiltfile
wait
