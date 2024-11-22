import os
import pickle
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Scopes nécessaires pour accéder à Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """
    Authentifie l'utilisateur et retourne le service API Google Drive.
    Assure une gestion claire des fichiers de credentials.
    """
    credentials_file = 'c:/documents/datascientest/examen/credentials.json'
    token_file = 'token.pickle'

    # Vérifie que le fichier credentials.json existe
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(
            f"Le fichier '{credentials_file}' est introuvable. Veuillez le placer dans le dossier du script."
        )

    creds = None

    # Charge les credentials existants s'ils sont disponibles
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    # Si les credentials ne sont pas valides ou inexistants, lance une nouvelle authentification
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        # Sauvegarde les nouveaux credentials
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    # Retourne le service Google Drive
    return build('drive', 'v3', credentials=creds)

def download_file(file_id, destination):
    """
    Télécharge un fichier depuis Google Drive.
    :param file_id: ID du fichier sur Google Drive.
    :param destination: Chemin où enregistrer le fichier téléchargé.
    """
    try:
        drive_service = authenticate_drive()
        request = drive_service.files().get_media(fileId=file_id)
        with io.FileIO(destination, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Téléchargement en cours... {int(status.progress() * 100)}%")
        print(f"Fichier téléchargé avec succès : {destination}")
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
