from os import system
#todo this would be replaced with users own keys using the api function "_get_user_package_keys"
system(
    # Clones Genie_API's Repo
    "git clone https://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@github.com/ruben970105/Genie_API.git &&"
    # Clones mini_Genie's Repo
    "git clone --branch Genie_Algo_Lab https://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@github.com/ruben970105/mini_Genie.git &&"
    # Clones Post_Processing_Genie's Repo
    "git clone https://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@github.com/ruben970105/Post_Processing_Genie.git &&"
    # Pip install mlfinlab
    "pip install https://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@raw.githubusercontent.com/hudson-and-thames-clients/mlfinlab/master/mlfinlab-1.6.0-py38-none-any.whl &&"
    # Clones vectorbt's Repo
    "pip install -U 'vectorbtpro[base] @ git+https://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@github.com/polakowo/vectorbt.pro.git'"
    # pip installs requirements
    "pip install -r mini_Genie/requirements.txt Post_Processing_Genie/requirements Genie_API/requirements  &&"
    # Run setup.py
    "python3 setup.py install"

    #R
)

