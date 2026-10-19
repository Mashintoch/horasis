# Releasing horasis

Run the checks in part 1 before every release, then publish with part 2.
Commands are PowerShell. On Linux or macOS, adjust the paths.

## 1. Testing

### Automated checks

From the repo root:

```powershell
pip install -e ".[dev,server]"
pytest -q
ruff check src tests
mypy
pylint $(git ls-files '*.py')
```

All four must pass, and CI must be green on `main` for Python 3.10, 3.11
and 3.12.

### Real models

The default test suite uses fake backends. Run the integration tests against
real weights, either from Actions (integration, then Run workflow) or
locally for whatever backends you have installed:

```powershell
pytest -m integration -v
```

Every backend should pass, or skip with a reason you understand.

### Package

```powershell
Remove-Item -Recurse -Force dist, build, src\horasis.egg-info -ErrorAction SilentlyContinue
python -m build
twine check dist/*
```

Install the wheel into a fresh venv, from outside the repo:

```powershell
cd C:\tmp
Remove-Item -Recurse -Force C:\tmp\hz -ErrorAction SilentlyContinue
python -m venv C:\tmp\hz
$whl = (Get-ChildItem C:\Users\Chibueze\dev\mashintoch-perceptospy-native\horasis\dist\*.whl).FullName

C:\tmp\hz\Scripts\pip install $whl
C:\tmp\hz\Scripts\horasis --version
C:\tmp\hz\Scripts\python -c "from horasis import Config; Config()"

C:\tmp\hz\Scripts\pip install "$whl[all]"
C:\tmp\hz\Scripts\python -c "import cv2, horasis; print(cv2.__version__, horasis.__version__)"
```

The core install must not pull in torch, ultralytics or opencv, and `[all]`
must resolve without errors.

### Smoke test each capability

In the `[all]` venv, fetch a test image, render one for OCR, and pull the
face model:

```powershell
cd C:\tmp
Invoke-WebRequest https://ultralytics.com/images/bus.jpg -OutFile bus.jpg
C:\tmp\hz\Scripts\python -c "from PIL import Image, ImageDraw, ImageFont; im=Image.new('RGB',(600,120),'white'); d=ImageDraw.Draw(im); d.text((20,40),'HORASIS OCR TEST 2026',fill='black',font=ImageFont.load_default(size=40)); im.save('text.png')"
C:\tmp\hz\Scripts\horasis models pull yunet
```

Then:

```powershell
$h = "C:\tmp\hz\Scripts\horasis"
& $h detect   bus.jpg --json --draw out_detect.jpg
& $h classify bus.jpg --json
& $h segment  bus.jpg --draw out_segment.jpg
& $h pose     bus.jpg --draw out_pose.jpg
& $h ocr      text.png --json
& $h ocr      text.png --backend tesseract --json
& $h faces    bus.jpg --draw out_faces.jpg
& $h moderate bus.jpg --json
& $h backends
```

| Command | Expected |
|---|---|
| detect | several `person` boxes and a `bus`, drawn on the right objects |
| classify | a bus or vehicle class on top, confidence between 0 and 1 |
| segment | masks over the people and the bus |
| pose | skeletons on the people |
| ocr | `HORASIS OCR TEST 2026`, give or take a character |
| faces | boxes and landmarks on the visible faces |
| moderate | `flagged: false`, scores between 0 and 1 |
| backends | every installed backend listed, no failed plugins |

Labels must be names (`person`), not indices (`0`), and boxes must stay
inside the image. If not, that backend's label mapping or preprocessing is
broken. The tesseract line needs the tesseract binary; skip it if you don't
have it.

### Access control

```powershell
C:\tmp\hz\Scripts\python -c @"
from horasis import Client, HashedApiKeyValidator, Tier
from horasis.foundation.errors import InvalidApiKeyError, RevokedApiKeyError

v = HashedApiKeyValidator()
key = v.issue(principal_id='alice', tier=Tier.FREE, expires_at=None, scopes=['detect'])
c = Client.build(validator=v)

print(len(c.detection.detect('bus.jpg', api_key=key).detections), 'detections')
print(c.usage(key))

try:
    c.detection.detect('bus.jpg', api_key='wrong')
except InvalidApiKeyError as e:
    print('invalid key:', e)

v.revoke(key)
try:
    c.detection.detect('bus.jpg', api_key=key)
except RevokedApiKeyError as e:
    print('revoked key:', e)
"@
```

Then check quotas. Copy `src/horasis/config/default.yaml` to
`C:\tmp\tiny.yaml`, set the free tier quota to 3, and build the client with
`Client.build(config=Config.from_yaml('tiny.yaml'), validator=v)`. The fourth
`detect` call must raise `QuotaExceededError` with `retry_after` set. A call
with `b'not an image'` must raise `ValidationError` and leave `usage()`
unchanged.

### Server

Terminal 1:

```powershell
$env:HORASIS_API_KEY = "change-me"
C:\tmp\hz\Scripts\horasis serve --port 8000
```

Terminal 2 (use `curl.exe`; in PowerShell `curl` is an alias for
`Invoke-WebRequest`):

```powershell
cd C:\tmp
curl.exe localhost:8000/health
curl.exe -H "X-API-Key: change-me" -F image=@bus.jpg localhost:8000/v1/detect
curl.exe -H "X-API-Key: change-me" localhost:8000/usage
curl.exe -i -H "X-API-Key: wrong" -F image=@bus.jpg localhost:8000/v1/detect
curl.exe -i -H "X-API-Key: change-me" -F image=@bus.jpg localhost:8000/v1/nope
```

The last two should return 401 and 404.

To check the SSRF guard, restart the server with URL inputs enabled and send
`http://127.0.0.1:8000/health` as the image URL. You should get a 422, and
the error must not mention the internal IP.

### Go / no-go

- [ ] CI green on `main`
- [ ] integration tests pass, or skip for known reasons
- [ ] wheel builds, core install stays slim
- [ ] every capability looks right on real images
- [ ] access control and server behave as above
- [ ] `CHANGELOG.md` has the new version and date
- [ ] README is accurate

## 2. Publishing

Releases go out through `.github/workflows/publish.yml` using PyPI trusted
publishing, so no API token is stored anywhere.

### One-time setup

1. Create accounts on https://pypi.org and https://test.pypi.org and turn on
   2FA for both.
2. Check https://pypi.org/project/horasis/ returns 404.
3. On both sites, add a pending trusted publisher (Account settings,
   Publishing):

   | Field | pypi.org | test.pypi.org |
   |---|---|---|
   | Project name | `horasis` | `horasis` |
   | Owner | `Mashintoch` | `Mashintoch` |
   | Repository | `horasis` | `horasis` |
   | Workflow | `publish.yml` | `publish.yml` |
   | Environment | `pypi` | `testpypi` |

4. In the GitHub repo settings, create the environments `pypi` and
   `testpypi`. Add yourself as a required reviewer on `pypi`.

### Dry run on TestPyPI

1. Set `__version__ = "1.0.0rc1"` in `src/horasis/__init__.py`. TestPyPI
   won't take the same version twice, so use `rc2`, `rc3` for retries.
2. Commit and push to `main`.
3. In Actions, run the publish workflow on `main`. A manual run only uploads
   to TestPyPI.
4. Install it:

   ```powershell
   python -m venv C:\tmp\hz-test
   C:\tmp\hz-test\Scripts\pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "horasis==1.0.0rc1"
   C:\tmp\hz-test\Scripts\horasis --version
   ```

5. Check the project page on test.pypi.org: README, links, classifiers and
   licence.

### Release

1. Set the final version and date the changelog entry.
2. Open a PR, wait for CI, merge.
3. Tag and push:

   ```powershell
   git switch main
   git pull
   git tag -a v1.0.0 -m "horasis 1.0.0"
   git push origin v1.0.0
   ```

4. The workflow builds, checks the tag against `__version__`, then waits for
   your approval on the `pypi` environment.
5. Install from PyPI in a fresh venv and run `horasis --version`.
6. Create a GitHub release from the tag with the changelog entry.

### Without Actions

```powershell
python -m build
twine check dist/*
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."
twine upload dist/*
```

Use a project-scoped API token and never commit it.

### When something goes wrong

- A broken release: yank it on pypi.org, fix it, release a patch version.
  Don't delete it.
- Version numbers can never be reused, even after deletion.
- "File already exists": bump the version.
- "invalid-publisher": the owner, repo, workflow or environment on PyPI
  doesn't match GitHub exactly.

### Versioning

Semantic versioning. Patch for fixes, minor for new features, major for
breaking changes to the public API (top-level exports, result schemas,
`Config`) or behaviour.
