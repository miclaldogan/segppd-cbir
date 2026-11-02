# CBIR Araştırma Raporu Yazım Promptu

> Bu promptu, projenin kök dizininde çalışan bir kodlama ajanına (ör. Claude Code) olduğu gibi verin. Yanına iki dosya ekleyin:
> 1. `CBIR_RESEARCH_REPORT.pdf`: üslup, yapı ve format için örnek rapor.
> 2. `build_report.js`: örnek raporu üreten betik. Format için şablon olarak kullanılacaktır.

---

## Görev

Bu projede yapılan içerik tabanlı görüntü erişimi (CBIR) çalışması için hocaya sunulacak, Türkçe ve gönderilmeye hazır bir **final araştırma raporu** yaz. Rapor, ekteki `CBIR_RESEARCH_REPORT.pdf` ile **aynı üslupta, aynı bölüm yapısında ve aynı biçimde** olmalıdır. İçerik tamamen bu projenin kendi veri setlerinden, kodundan ve kaydedilmiş sonuçlarından gelmelidir. Örnek rapordaki sayıları, veri setlerini ya da bulguları **kesinlikle kopyalama**. Örnek rapor yalnızca üslup ve format referansıdır.

Çıktılar:
- `docs/CBIR_RESEARCH_REPORT.docx` ve `docs/CBIR_RESEARCH_REPORT.pdf`
- `report/build_report.js`: raporu üreten betik
- `report/make_figures.py`: şekilleri üreten betik
- `report/verify_numbers.py`: rapordaki her sayıyı kaydedilmiş sonuç dosyalarıyla karşılaştıran betik
- `docs/REPORT_PROVENANCE.md`: hangi tablo, şekil ve sayının hangi dosyadan geldiğini gösteren kısa liste

## 1. Önce projeyi incele, sonra yaz

1. Kodu, konfigürasyonları, veri bölme dosyalarını, metrik çıktılarını (CSV/JSON), logları ve mevcut notları oku. Hangi veri setlerinin, modellerin, loss fonksiyonlarının ve değerlendirme protokollerinin kullanıldığını çıkar.
2. Rapordaki **her sayı** kaydedilmiş bir sonuç dosyasından gelmelidir. Bir sayıyı bulamıyorsan uydurma ve tahmin etme. Eksik olanı "hesaplanmamıştır" diye belirt ya da bana sor.
3. Raporu yazmak için yeni bir model eğitme. Sonuçlar eksikse önce bana listele.
4. Kesin karar veremediğin şeyleri yazmaya başlamadan önce kısa bir soru listesiyle bana sor. Örnekler: hangi model "önerilen model", hangi küme kilitli test, veri setinin lisansı, hocanın verdiği referans yöntem.

## 2. Yazım üslubu (en önemli kısım)

Örnek rapordaki dili birebir uygula:

- **Edilgen ve nesnel anlatım kullan.** "-mıştır / -ilmiştir / -mektedir / -maktadır" kalıplarını kullan. Örnekler: "Arama tam Öklid mesafesiyle yapılmıştır.", "Bu durum ... göstermektedir."
- **Asla birinci şahıs kullanma.** "Biz, denedik, bulduk, yaptık, amacımız" yazma. Hocaya doğrudan atıf da yapma ("hocamızın yöntemi" yazma). Hocanın önerdiği yöntem için "referans yöntem" de.
- **Teknik terimleri İngilizce bırak.** Türkçe ek kesme işaretiyle eklenir: embedding'ler, backbone'un, fine-tune edilmiş, seed'ler, feature'ları. Kullanılacak terimler: embedding, backbone, feature, fine-tuning, projection head, loss, batch, epoch, learning rate, optimizer, scheduler, early stopping, patience, augmentation, dropout, token, patch, pooling, ensemble, seed, whitening, re-ranking, query expansion, knowledge distillation, domain shift, data leakage, closed-set, precision-recall, average precision, macro/micro, forward pass, offline.
- Yerleşmiş Türkçe kelimeleri Türkçe bırak: sorgu, galeri, veri seti, sınıf, eğitim, doğruluk, güven aralığı, ölçüt, ağırlık, katman, mesafe, kilitli test, geliştirme (dev) kümesi, keşifsel. "Gömme, omurga, ince ayar, tohum, topluluk, jeton, yama, kesinlik-anma, beyazlatma" gibi zorlama çeviriler **kullanma**.
- **Kısa, net ve teknik cümleler kur.** Her cümlede tek bir bilgi ver. Sayıyı cümlenin içinde ver: "Hit@1 0,807 [0,756; 0,856] olarak ölçülmüştür."
- **İddiaları temkinli yaz.** Güven aralığı bir eşiği içeriyorsa "hedefe ulaşıldığı istatistiksel olarak gösterilememiştir" de. Anlamlı olmayan kazanca "kazanç" deme. Nedeni kanıtlanamayan bir durum için "düşündürmektedir" ya da "olası nedenler şunlardır: (i) ..., (ii) ..." kalıbını kullan. "En iyi sonuç" ya da "state-of-the-art" iddiasında bulunma. Literatürdeki sayıların neden doğrudan karşılaştırılamadığını açıkla.
- **Olumsuz bulguları sakla.** İşe yaramayan denemeleri ayrı bir alt bölümde, hangi yönlerin denendiğini göstermek için raporla.
- **Sınırlılıkları açıkça yaz.** Kapsam dışı kalan sınıfları, küçük test kümelerini, etiket gürültüsünü, harici genellemeyi ve ilgililik tanımını numaralı ve kalın başlıklı maddeler hâlinde ver: "(i) **Kapsam:** ...".
- Rapor metninde dosya yolu (`docs/...`, `artifacts/...`), "TODO", "kontrol edilmeli", "otomatik özetlendi" gibi iç notlar ve "sonradan yapıldı" ifadesi bulunmamalıdır.
- Sayı biçimi Türkçe olmalıdır: ondalıkta virgül (0,807), binlerde nokta (1.485), güven aralıkları `[0,756; 0,856]` biçiminde. Ölçütlerde 3 ondalık basamak kullan.

## 3. Bölüm yapısı

Örnek raporla aynı sırayı ve başlık adlarını kullan. Bir bölüm bu projeye uymuyorsa çıkar, ama sırayı bozma.

- **Başlık**: "<Veri seti> Üzerinde İçerik Tabanlı <alan> Görüntü Erişimi: <Referans yöntem>'den <Önerilen model>'e". Altında "Öğrenci No - Ad Soyad" ve "Araştırma Raporu - <Ay Yıl>" satırları yer alır. Ad, numara ve tarihi bana sor.
- **Özet**: Tek paragraf, yaklaşık 10-12 cümle. Problem, referans yöntem, önerilen model, data leakage önlemi, ana sayılar (güven aralığıyla), harici sonuç, işe yaramayanlar ve kapsam sınırı yer alır. Ardından **Anahtar Kelimeler** satırı gelir.
- **1. Giriş ve Araştırma Problemi**: Uygulama problemi; CBIR'in sınıflandırmadan farkı (çıktı bir sıralamadır); referans yöntem; erişim adımının sabit tutulup feature uzayının iyileştirildiği; amaç.
- **2. Veri Setleri**: Her veri seti için ayrı alt başlık. Kaynak, DOI, lisans, yerel denetimde bulunan görüntü/sınıf sayısı, çözünürlük ve biçim. Makaledeki sayılarla tutarsızlık varsa bunu yaz. Kopya ve near-duplicate taraması (hash yöntemi ve eşikler), çıkarılan görüntüler, değerlendirmeye alınan sınıflar ve dışarıda kalanların nedeni. Örnek görüntü şekli ve sınıf dağılımı şekli. Harici veri seti varsa cross-dataset leakage taraması ve sınıf eşlemesi.
- **3. İlgili Çalışmalar**: Tablo 1, sütunlar örnek rapordaki gibi olmalı: Çalışma | Görev | Veri (sınıf) | Yöntem | Ölçüt ve sonuç | Karşılaştırılabilir mi? Ardından neden doğrudan karşılaştırılamadığını anlatan bir paragraf gelir. **Yalnızca gerçekten var olan ve doğrulayabildiğin kaynakları** kullan. DOI ya da arXiv numarası yazmalısın. Erişemediğin bir çalışmayı tabloya alma ve bunu metinde belirt.
- **4. Yöntem ve Mimari**: Denemeleri yapılış sırasına göre alt bölümlerde anlat: referans yöntem, eğitimsiz iyileştirmeler, eğitimli modeller, önerilen nihai mimari. Önerilen mimari için tüm ayrıntıları ver: backbone, dondurulan ve eğitilen bloklar, eğitilebilir parametre sayısı, learning rate'ler, embedding boyutu, batch yapısı, augmentation'lar, optimizer, scheduler, early stopping kuralı, loss ve hiperparametreleri, çıkarım adımları. Referans mimari ve önerilen mimari için birer şema şekli çiz; önerilen mimaride eğitim ve çıkarım parçalarını ayır. Ek denemeleri (mesafe metriği, füzyon, re-ranking, parametre azaltma vb.) son alt bölümde kısaca tanıt ve "ayrıntıları Bölüm 6.x'te verilmiştir" de.
- **5. Değerlendirme Protokolü ve Ölçütler**: İlgililik tanımı (aynı sınıf etiketi; tarımsal ya da uzman teşhisi değildir). Protokol tablosu: Protokol | Sınıf | Eğitim | Galeri | Sorgu | Not. Hit@K, P@K, R@K, AP/mAP, macro/micro, bootstrap güven aralığı, P-R eğrisi, ROC-AUC ve F1'in nasıl hesaplandığı tanım olarak verilir.
- **6. Deneysel Sonuçlar ve Ablation Çalışması**: **Tüm deneyler bu bölümde, tek bir akış hâlinde verilir; ayrı bir "Ekler" bölümü olmamalıdır.** Bölümün başında alt bölümlerin sırası ve hangi sonuçların keşifsel olduğu açıklanır. Önerilen sıra: eğitimsiz referans; olumsuz bulgular; kilitli testte ablation (tablo + çubuk grafik + P@K/R@K + P-R/ROC); loss karşılaştırması; mesafe metrikleri ve füzyon; re-ranking; parametre azaltma ya da verimlilik. Projede yapılmayan deneylerin alt bölümünü çıkar.
- **7. Nihai Model ve Sonuçlar**: Önerilen modelin tüm test kümelerindeki sonuç tablosu (güven aralığıyla), referans yöntemle karşılaştırma tablosu, sınıf başına başarı şekli, rastgele seçilmiş (sabit seed, elle seçilmemiş) başarılı ve başarısız sorgu örnekleri, iç ve harici farkın olası nedenleri, hedef değer hakkında temkinli bir yorum, maliyet (parametre, eğitim süresi, GPU belleği, sorgu başına arama süresi).
- **8. Sonuç ve Sınırlılıklar**: Bir sonuç paragrafı ve numaralı sınırlılıklar paragrafı. Sonraki adımlar kısaca belirtilir.
- **Kaynaklar**: `[1] Yazar, A., Yazar, B. Başlık. Dergi, cilt(sayı):sayfa, yıl. doi:...` biçiminde, metinde ilk geçiş sırasına göre numaralanır. Kaynaklar raporun en sonunda yer alır.

Tablo ve şekiller görünme sırasına göre numaralanır (Tablo 1, 2, ...; Şekil 1, 2, ...). Metinde her tablo ve şekle atıf yapılır ("Tablo 4'te görüldüğü gibi ...").

## 4. Biçim

Ekteki `build_report.js` betiğini şablon olarak kullan. Yardımcı fonksiyonları (`P`, `H1`, `H2`, `CAP`, `CAPT`, `FIG`, `TABLE`, `SP`) ve doküman ayarlarını **değiştirmeden** al, yalnızca içeriği yeniden yaz. Bu ayarlar şunlardır:

- A4 sayfa; kenar boşlukları üstte ve altta 2 cm, solda ve sağda 2,3 cm. Font Arial.
- Gövde metni 10 pt, iki yana yaslı, satır aralığı yaklaşık 1,1. H1 13 pt kalın, H2 11 pt kalın.
- Tablo başlıkları tablonun üstünde, şekil başlıkları şeklin altında yer alır; 8 pt, italik, ortalı.
- Tablolar ince gri kenarlıklı olur, başlık satırı açık mavi (`E8EEF4`), önerilen modelin satırı açık yeşil (`EAF4E8`) ve kalındır. Tablo yazısı 7,5-8,5 pt'dir.
- Sayfa numarası alt ortada yer alır. Kaynaklar 8,5 pt, asılı girintiyle yazılır.
- Paragraf içinde vurgu için `**kalın**` kullanılır (betik bunu destekler). Metinde ters tırnak (`) kullanılmaz.
- Grafikler matplotlib ile üretilir; şekillerin içindeki yazılar da aynı terminolojiyle (Türkçe + İngilizce teknik terimler) yazılır.
- DOCX, LibreOffice ile PDF'e çevrilir: `soffice --headless --convert-to pdf`.

## 5. Bitirmeden önce kontrol et

1. `verify_numbers.py` çalıştırılır ve metindeki tüm ana sayılar kaydedilmiş dosyalarla eşleşmelidir. Eşleşmeyen sayı varsa düzelt.
2. PDF metninde `biz`, `yaptık`, `denedik`, `hocamız`, `gömme`, `tohum`, `topluluk`, `ince ayar`, `TODO`, `docs/`, `artifacts/` ve `Ek A` araması yapılır; hiçbiri bulunmamalıdır.
3. Tablo ve şekil numaralarının sıralı olduğu, metindeki her atfın doğru tabloya ya da şekle gittiği kontrol edilir.
4. PDF sayfaları görsel olarak incelenir: taşan tablo, kırılmış şekil ya da boş sayfa olmamalıdır.
5. Raporda veri setinden fotoğraf varsa ve görüntülerin hakları belirsizse, şekil başlığına "yalnızca akademik değerlendirme amacıyla gösterilmiştir" notu eklenir.
6. Bitince bana kısa bir özet ver: hangi sayılar doğrulandı, hangi bilgiler eksik kaldı ya da varsayıma dayanıyor, hangi kaynaklar doğrulanamadı.
