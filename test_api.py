"""
test_api.py — Script de test de l'API Colon Cancer
Usage: python3 test_api.py [--host http://localhost:8000]
"""

import sys, json, time, argparse, urllib.request, urllib.error

def req(method, url, data=None):
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def ok(label, cond, detail=""):
    icon = "✅" if cond else "❌"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return cond

def section(title):
    print(f"\n{'─'*50}\n  {title}\n{'─'*50}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()
    BASE = args.host.rstrip("/")
    print(f"\n🧬 Colon Cancer API — Test Suite\n   Target: {BASE}")
    passed = failed = 0

    section("1. Health check")
    status, body = req("GET", f"{BASE}/health")
    r = ok("HTTP 200", status == 200)
    r &= ok("status = ready", body.get("status") == "ready", body.get("status"))
    r &= ok("model_loaded = True", body.get("model_loaded") is True)
    passed += r; failed += not r

    section("2. Top genes")
    status, body = req("GET", f"{BASE}/top-genes?n=10")
    r = ok("10 gènes retournés", isinstance(body, list) and len(body) == 10)
    if isinstance(body, list) and body:
        r &= ok("1er gène = M63391", body[0]["gene"] == "M63391", body[0]["gene"])
    passed += r; failed += not r

    section("3. Model info")
    status, body = req("GET", f"{BASE}/model-info")
    r = ok("HTTP 200", status == 200)
    if status == 200:
        r &= ok("test_accuracy > 0.70", body["metrics"].get("test_accuracy", 0) > 0.70,
                str(body["metrics"].get("test_accuracy")))
        r &= ok("auc_roc > 0.80", body["metrics"].get("test_auc_roc", 0) > 0.80,
                str(body["metrics"].get("test_auc_roc")))
    passed += r; failed += not r

    section("4. Prediction")
    import random; random.seed(42)
    gene_vec = [random.uniform(1000, 15000) for _ in range(2001)]
    status, body = req("POST", f"{BASE}/predict", {"gene_expression": [1.0]*100})
    r = ok("422 si mauvais nb features", status == 422)
    passed += r; failed += not r

    status, body = req("POST", f"{BASE}/predict", {"gene_expression": gene_vec})
    r = ok("HTTP 200 avec 2001 features", status == 200)
    if status == 200:
        r &= ok("prediction in [Normal, Abnormal]",
                body["prediction"] in ["Normal","Abnormal"], body.get("prediction"))
        r &= ok("probabilités somment à ~1.0",
                abs(body.get("probability_normal",0)+body.get("probability_abnormal",0)-1.0) < 0.01)
        print(f"\n     → {body['prediction']} (P(Abnormal)={body['probability_abnormal']:.3f}, confiance={body['confidence']})")
    passed += r; failed += not r

    section("5. Latence")
    times = [(lambda t0: (time.time()-t0)*1000)(time.time()) or req("POST",f"{BASE}/predict",{"gene_expression":gene_vec}) for _ in range(5)]
    # proper latency test
    times = []
    for _ in range(5):
        t0 = time.time()
        req("POST", f"{BASE}/predict", {"gene_expression": gene_vec})
        times.append((time.time()-t0)*1000)
    avg = sum(times)/len(times)
    r = ok(f"Latence moyenne < 200ms", avg < 200, f"{avg:.0f}ms")
    passed += r; failed += not r

    print(f"\n{'═'*50}")
    print(f"  Résultats : {passed}/{passed+failed} tests passés")
    print("  🎉 Tous les tests sont passés !" if failed==0 else f"  ⚠️  {failed} test(s) échoué(s)")
    print(f"{'═'*50}\n")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
