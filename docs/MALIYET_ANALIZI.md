# Mimari maliyet analizi: "topluluk + ResNet50 füzyonu" ile normal tek model

Kod: `src/segppd_cbir/rv2/cost_study.py`, tablo: `artifacts/research_v3/cost/cost_comparison.csv`.
Yöntem: parametre ve ileri geçiş MAC sayısı mimarilerden sayıldı (CPU, rastgele başlangıç; DINOv2-B 224 px, ResNet50 256 px, tek görüntü, çevirme ortalaması hariç). GPU gömme süreleri (DINOv2-B 10,6 ms, ResNet50 10,0 ms) ve eğitim süreleri (DINOv2-B 600-770 s, ResNet50 660 s) önceki ölçümlerden alındı ve **toplandı**; birleşik hat olarak yeniden ölçülmedi. Arama süresi bu oturumda eğitim çalışırken CPU'da ölçüldüğü için gürültülü; temiz ölçüm olarak eski değerler (256 boyut 0,26 ms, 768 boyut 0,75 ms, 809 görüntülük galeri) esas alındı.

| | Tek DINOv2-B + başlık (normal model) | 3 tohum topluluğu | Topluluk + ResNet50 füzyonu | **Tek DINOv2-B + ResNet50 füzyonu** (eklendi 2026-10-26) |
|---|---|---|---|---|
| Parametre (çıkarım için gerekli) | 85,9 M | 257,8 M (3,0x) | 281,8 M (3,3x) | 110,0 M (1,3x) |
| fp32 ağırlık | 344 MB | 1,03 GB | 1,13 GB | 440 MB |
| İleri geçiş (görüntü başına) | 21,9 GMAC | 65,8 (3,0x) | 71,2 (3,2x) | 27,3 (1,2x) |
| Gömme süresi (GPU, çevirme dahil, toplam) | 10,6 ms | 31,8 ms | 41,8 ms (3,9x) | 20,6 ms (1,9x) |
| Eğitim süresi (toplam) | 10-13 dk | 30-39 dk | 41-50 dk (3,8-4,1x) | 21-24 dk (2x) |
| Tanımlayıcı boyutu / 100 bin görüntü | 256-B, 102 MB | 768-B, 307 MB | 1024-B, 410 MB (4x) | 512-B, 205 MB (2x) |
| Tam Öklid arama (809 görüntü, temiz ölçüm) | 0,26 ms | 0,75 ms | ~1 ms (ölçülmedi, tahmin) | ~0,5 ms |

Doğruluk (Protokol B testi, Hit@1; keşifsel): tek Proxy-Anchor 0,825, tek SupCon 0,807, SupCon topluluğu 0,822, Proxy-Anchor topluluğu 0,836, SupCon topluluğu + ResNet50 0,840. Rapordaki eşleştirilmiş sınamada topluluğun ve füzyonun kazancı anlamlı değildir (füzyon, topluluğa göre +0,018 [−0,004; +0,044]).

## Değerlendirme
- **Ağır kısım topluluk (3x), ResNet50 değil.** ResNet50 hesap olarak yalnızca +%8 MAC ekler, ama ölçülen gömme süresine +%31 (10,0 ms) ve eğitim süresine +%28-37 (660 s) ekler. Topluluk, tek modele göre ~+1 ile +1,5 puan veriyor ve anlamlı değil.
- **Topluluk büyük ölçüde gereksiz.** Daha iyi kayıpla (Proxy-Anchor) eğitilmiş tek model (0,825), eski SupCon topluluğundan (0,822) maliyetin üçte birinde aynı düzeyde. Topluluk, zayıf kaybın telafisiydi.
- **Füzyon (+ResNet50) topluluğa göre +%8 MAC, +%31 gömme süresi, +%28-37 eğitim süresi karşılığında +1,8 puan**, ama güven aralığı sıfırı içeriyor ve ResNet50 tek başına 0,511. Mimari karmaşıklığı (iki farklı ağ, iki ön işleme) artırıyor.
- **Mevcut ölçekte hesap sorun değil:** 809 görüntülük galeride bile 42 ms/sorgu, 100 bin görüntüde indeks 410 MB. Yani ağır ama kullanılamaz değil. Ağırlık, sorgu tarafında 3,9x gecikme (94 yerine 24 görüntü/sn, RTX 4050) ve 3,3x bellek, galeri kurulumunda 4x süre.
- **Sonuç:** Gerekli değil. Tek DINOv2-B + Proxy-Anchor (referans aday) ile aynı sonuç aralığındayız. Topluluk/füzyon yalnızca son 1-3 puanı kovalamak için anlamlı, ve bu kazanç şu an istatistiksel olarak doğrulanmadı.
- **Tek model + ResNet50 füzyonu (eklendi):** Proxy-Anchor tek model + ResNet50 (w=0,05, Öklid) B testinde 0,839, C testinde 0,800 (keşifsel; `docs/DISTANCE_METRIC_STUDY.md`). Topluluk füzyonunun (0,840 / 0,773) sonuç aralığında, hesabın %38'i ile. Topluluk + beyazlatma 0,858'in tek füzyona göre farkı B'de +0,019 [−0,015; +0,053], C'de −0,040 [−0,102; +0,018]: topluluk gereksiz. Füzyonun tek modele katkısı +0,013 [+0,003; +0,025] (B) ve +0,018 [−0,004; +0,040] (C), taze testle doğrulanmalı.
- **Denenmeyen ucuz alternatifler:** bilgi damıtma (topluluktan tek modele), DINOv2-S (daha hafif, B testinde ~0,72-0,76 ile belirgin daha düşük).
