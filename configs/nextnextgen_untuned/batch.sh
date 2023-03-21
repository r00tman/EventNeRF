#!/usr/bin/env bash
for scene in bottle chicken controller cube dragon microphone multimeter plant tapes; do
    echo $scene
    cp _template.txt $scene.txt
    sed -i s/_template/$scene/g $scene.txt
    cp ../../scripts/job_nextnextgen_template.sh ../../scripts/job_nextnextgen_$scene.sh
    sed -i s/_template/$scene/g ../../scripts/job_nextnextgen_$scene.sh
done
