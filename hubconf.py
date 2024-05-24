dependencies = ['torch', 'torchaudio']

def enhance(pretrained = True):

    # Imports
    import torch
    from supervoice_enhance.wrapper import SuperVoiceSeparate
    from supervoice_enhance.config import config

    # Model
    vocoder = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vocoder', model='bigvsan')
    flow = torch.hub.load(repo_or_dir='ex3ndr/supervoice-flow', model='flow')
    model = SuperVoiceSeparate(flow, vocoder)

    # Load checkpoint
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice-separate-60000.pt", map_location="cpu")
        model.diffusion.load_state_dict(checkpoint['model'])

    return model
            