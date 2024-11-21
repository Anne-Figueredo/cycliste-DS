import os
import pickle
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Scopes nécessaires pour accéder à Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """Authentifie l'utilisateur et retourne le service API."""
    creds = None
    # Le fichier token.pickle stocke les informations d'authentification de l'utilisateur.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Si il n'y a pas de credentials valides, demande l'authentification à l'utilisateur.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Sauvegarde les credentials pour la prochaine exécution
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Retourne le service Google Drive
    return build('drive', 'v3', credentials=creds)

def download_file(file_id, destination):
    """Télécharge un fichier depuis Google Drive."""
    drive_service = authenticate_drive()
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    print(f'Fichier téléchargé à : {destination}')
