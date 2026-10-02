import sys, time, nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": sys.argv[3]}})
t=time.time()
try:
    client.execute()
    print("COMPLETED OK", time.time()-t)
except CellExecutionError as e:
    print("FAILED", time.time()-t); print(str(e)[-3000:])
finally:
    nbformat.write(nb, dst)
