import os
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET

# Jenkins URL and credentials
jenkins_url = 'http://localhost:8090/'
username = 'admin'
api_token = '123456'

# Directory to save Jenkinsfiles
output_dir = 'jenkinsfiles'
os.makedirs(output_dir, exist_ok=True)

# Function to get Jenkins job configuration
def get_job_config(job_url):
    config_url = f'{job_url}config.xml'
    response = requests.get(config_url, auth=HTTPBasicAuth(username, api_token))
    response.raise_for_status()
    return ET.fromstring(response.content)


# Get list of all jobs
jobs_url = f'{jenkins_url}/api/xml?tree=jobs[name,url]'
response = requests.get(jobs_url, auth=HTTPBasicAuth(username, api_token))
response.raise_for_status()
jobs = ET.fromstring(response.content)

for job in jobs.findall('job'):
    job_name = job.find('name').text
    job_url = job.find('url').text
    try:
        config_xml = get_job_config(job_url)
    except requests.exceptions.RequestException as e:
        print(f"Failed to get config for job {job_name}: {e}")
        continue

    # Check if it's a pipeline job
    definition = config_xml.find('definition')
    if definition is not None:
        if definition.get('class') == 'org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition':
            # Jenkinsfile is defined inline
            script = definition.find('script').text
            filename = os.path.join(output_dir, f'{job_name}_Jenkinsfile')
            try:
                with open(filename, 'w') as file:
                    file.write(script)
                print(f'Saved Jenkinsfile for job {job_name} as {filename}')
            except IOError as e:
                print(f"Failed to save Jenkinsfile for job {job_name}: {e}")
        elif definition.get('class') == 'org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition':
            # Jenkinsfile is defined in SCM
            scm = definition.find('scm')
            user_remote_configs = scm.find('userRemoteConfigs')
            url = user_remote_configs.find('hudson.plugins.git.UserRemoteConfig/url').text
            branches = scm.find('branches')
            branch = branches.find('hudson.plugins.git.BranchSpec/name').text
            script_path = definition.find('scriptPath').text
            print(f'Job Name: {job_name}')
            print('Jenkinsfile (SCM):')
            print(f'Repository URL: {url}')
            print(f'Branch: {branch}')
            print(f'Script Path: {script_path}')
            print('--------------------------------------------------------')
    else:
        print(f"Job {job_name} does not have a Pipeline definition.")
