"""Exercise the server-check client over a local HTTP connection."""
import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

import numpy as np
from plyfile import PlyData,PlyElement

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from verify_server_reconstruction import exercise


class Handler(BaseHTTPRequestHandler):
    image=b'test image bytes'
    wrong_hash=False
    nonfinite=False
    missing_fields=False
    loaded=False
    def log_message(self,*args):pass
    def send_bytes(self,data,content_type='application/json'):
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        if content_type=='application/octet-stream':
            self.send_header('X-Idol-Metadata','/v1/reconstructions/fixture/metadata')
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':
            self.send_bytes(json.dumps({'model_loaded':self.loaded}).encode())
        else:
            checksum='0'*64 if self.wrong_hash else hashlib.sha256(self.image).hexdigest()
            self.send_bytes(json.dumps({'input':'uploaded_image','image_sha256':checksum}).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.__class__.loaded=True
        names=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        if self.missing_fields:names.remove('rot_3')
        points=np.zeros(32,dtype=[(name,'f4') for name in names])
        if self.nonfinite:points['x'][0]=np.nan
        output=io.BytesIO();PlyData([PlyElement.describe(points,'vertex')],text=False).write(output)
        self.send_bytes(output.getvalue(),'application/octet-stream')


class ServerVerification(unittest.TestCase):
    def setUp(self):
        Handler.wrong_hash=False;Handler.nonfinite=False;Handler.missing_fields=False;Handler.loaded=False
        self.directory=tempfile.TemporaryDirectory()
        self.image=Path(self.directory.name)/'input.bin';self.image.write_bytes(Handler.image)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.thread.join();self.server.server_close();self.directory.cleanup()
    def test_valid_cloud_and_matching_metadata(self):
        report,payload=exercise(self.url,self.image,gaussians=32)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['point_count'],32)
        self.assertEqual(report['point_cloud_sha256'],hashlib.sha256(payload).hexdigest())
    def test_wrong_input_identity_is_refused(self):
        Handler.wrong_hash=True
        with self.assertRaisesRegex(ValueError,'metadata'):exercise(self.url,self.image,gaussians=32)
    def test_nonfinite_cloud_is_refused(self):
        Handler.nonfinite=True
        with self.assertRaisesRegex(ValueError,'non-finite'):exercise(self.url,self.image,gaussians=32)
    def test_missing_gaussian_field_is_refused(self):
        Handler.missing_fields=True
        with self.assertRaisesRegex(ValueError,'Gaussian properties'):exercise(self.url,self.image,gaussians=32)

if __name__=='__main__':unittest.main()
