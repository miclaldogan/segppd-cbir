// Source text of docs/CBIR_RESEARCH_REPORT.docx (Turkish). Run: NODE_PATH=<docx install> node report/build_report.js
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, BorderStyle,
  Footer, PageNumber, HeadingLevel, VerticalAlign } = require('docx');
const D = JSON.parse(fs.readFileSync('report/data.json'));
const FONT = 'Arial';
const f3 = (x) => x.toFixed(3).replace('.', ',');

function runs(text, base = {}) {            // **bold** inline markup
  return text.split('**').map((t, i) => new TextRun({ text: t, font: FONT, size: 20, bold: i % 2 === 1, ...base }));
}
const P = (text, opt = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 110, line: 270 }, ...opt });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true })], spacing: { before: 260, after: 110 }, keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT, size: 22, bold: true })], spacing: { before: 180, after: 90 }, keepNext: true });
const CAP = (t) => new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 16, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 40, after: 180 } });
const CAPT = (t) => new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 16, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true });
function FIG(name, widthPx, caption) {
  const d = D.figs[name]; const w = widthPx, h = Math.round(widthPx * d.h / d.w);
  return [new Paragraph({ children: [new ImageRun({ type: 'png', data: fs.readFileSync(`report/figs/${name}.png`), transformation: { width: w, height: h }, altText: { title: name, description: caption, name } })],
    alignment: AlignmentType.CENTER, spacing: { before: 100, after: 0 }, keepNext: true }), CAP(caption)];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: 'BBBBBB' };
const borders = { top: border, bottom: border, left: border, right: border };
function TABLE(widths, header, rows, opt = {}) {
  const total = widths.reduce((a, b) => a + b, 0); const fs_ = opt.size || 17;
  const cell = (t, w, hdr, left, bold) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: hdr ? { fill: 'E8EEF4', type: ShadingType.CLEAR, color: 'auto' } : (bold ? { fill: 'EAF4E8', type: ShadingType.CLEAR, color: 'auto' } : undefined),
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ alignment: left ? AlignmentType.LEFT : AlignmentType.CENTER, children: [new TextRun({ text: String(t), font: FONT, size: fs_, bold: hdr || bold })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, widths[i], true, i === 0)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false, i < (opt.leftCols || 1), opt.boldRows && opt.boldRows.includes(ri))) }))] });
}
const SP = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- data-driven tables
const B = D.B, C = D.C, ex = D.ext, pl = D.pooled, L = D.lesion;
const tB = B.map((r) => [r.name, f3(r.hit1), f3(r.p10), f3(r.hit10), f3(r.mAP), f3(r.f1), f3(r.auc)]);
const best = B[8], bestC = C[8];
const tFinal = [
  ['SegPPD-101, Protokol B (kilitli test)', '225', '75', `${f3(best.hit1)} [0,756; 0,856]`, f3(best.p10), f3(best.hit10), f3(best.mAP), f3(best.f1), f3(best.auc)],
  ['SegPPD-101, Protokol C (ikinci kilitli test)', '142', '75', `${f3(bestC.hit1)} [0,689; 0,840]`, f3(bestC.p10), f3(bestC.hit10), f3(bestC.mAP), f3(bestC.f1), f3(bestC.auc)],
  ['Protokol B + C birleşik (galeriler farklı)', '367', '75', `${f3(pl.hit1)} [${f3(pl.lo)}; ${f3(pl.hi)}]`, '-', '-', '-', '-', '-'],
  ['Harici: PlantSeg (sızıntısız)', '1.485', '11 (arama 75 sınıf içinde)', `${f3(ex.hit1)} [${f3(ex.lo)}; ${f3(ex.hi)}]`, f3(ex.p10), f3(ex.hit10), '-', '-', '-']];

const kids = [];
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 70 }, children: [new TextRun({ text: 'SegPPD-101 Üzerinde İçerik Tabanlı Bitki Hastalığı Görüntü Erişimi: Donmuş ResNet50\'den DINOv2 Tabanlı Önerilen Modele', font: FONT, size: 28, bold: true })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: 'Araştırma Raporu - Ekim 2026', font: FONT, size: 20 })] }));

// ================= 1
kids.push(H1('1. Giriş ve Araştırma Problemi'));
kids.push(P('Bir bitki yaprağında gördüğümüz lekenin hangi hastalığa ya da zararlıya ait olduğunu anlamak, çoğu zaman uzman bilgisi gerektirir. Elimizdeki görüntüye en çok benzeyen, etiketi bilinen örnekleri hızlıca getirebilen bir sistem, hem uzmana hem de çiftçiye karşılaştırma imkânı sunar. Ancak gerçek tarla görüntüleri bu iş için zordur: arka plan, ışık, çekim mesafesi ve kadraj çok değişkendir, aynı hastalık farklı evrelerde farklı görünür, farklı hastalıklar ise birbirine çok benzeyebilir.'));
kids.push(P('İçerik tabanlı görüntü erişimi (CBIR) ile sıradan sınıflandırma arasındaki fark şudur: sınıflandırmada model bir görüntü için tek bir etiket üretir; CBIR\'de ise görüntüler bir öznitelik uzayına yerleştirilir ve sorguya en yakın galeri görüntüleri sıralanarak getirilir. Sistemin çıktısı bir etiket değil, bir sıralamadır; bu yüzden ölçütlerimiz de Hit@K, Precision@K ve mAP gibi sıralama ölçütleridir. Bu çalışmada, hocamızın önerdiği donmuş ResNet50 + tam Öklid sıralama yöntemini temel aldık ve erişim adımını değiştirmeden öznitelik uzayını iyileştirmeye çalıştık. Amacımız, gerçek ortam görüntülerinde Hit@1 değerini mümkün olduğunca yükseltmek ve bunu dürüst, sızıntısız bir protokolle ölçmektir.'));

// ================= 2
kids.push(H1('2. Veri Setleri'));
kids.push(H2('2.1 SegPPD-101'));
kids.push(P('SegPPD-101 (Zenodo, DOI 10.5281/zenodo.15114160, MIT lisansı) [7], bitki yaprağı hastalık ve zararlıları için hazırlanmış, az örnekli semantik segmentasyon (few-shot segmentation) veri setidir [1]. Yerel denetimimizde 2.263 görüntü ve 2.263 maske, tamamı 256×256 piksel RGB PNG olarak bulundu; 101 orijinal sınıf (bitki_durum biçiminde, 27 bitki türü) vardır ve bu sınıfların 27\'si sağlıklı yaprak sınıfıdır. Maske, açıklanan hedef bölgeyi gösterir; leke hastalıklarında lekeyi, yaygın hastalıklarda ve sağlıklı yapraklarda ise tüm yaprağı kapsar (Şekil 1). Biz veri setini segmentasyon için değil, görüntü erişimi için kullandık; maskeleri yalnızca hedef bölge büyüklüğüne göre analiz yapmak için kullandık, hiçbir erişim sisteminde girdi olarak vermedik. Orijinal makale bitki türü sayısını bir yerde 27, başka bir yerde 21 olarak veriyor [1]; yerel denetimimiz 27 türü doğruladı.'));
kids.push(...FIG('dataset_samples', 580, 'Şekil 1. SegPPD-101\'den örnek görüntüler (üst satır) ve hedef bölge maskeleri (alt satır, kırmızı kaplama). Görüntüler yalnızca akademik değerlendirme amacıyla gösterilmiştir; alt kaynakların hakları belgelenmediği için yeniden dağıtılmamalıdır.'));
kids.push(P('**Değerlendirdiğimiz sınıf sayısı 101 değil, 75\'tir.** Şekil 2\'de görüldüğü gibi sınıflar dengesizdir: kopyalar çıkarıldıktan sonra kalan 2.220 görüntü içinde sınıf başına 7 ile 67 arasında görüntü vardır (medyan 21). Sorgu, geliştirme (dev), eğitim ve galeri kümelerini birbirinden ayrık kurabilmek için bir sınıfta en az 13 kullanılabilir görüntü şartı koyduk (Protokol A için 3 sorgu + 10 galeri; Protokol B için 3 test + 2 dev sorgusu ve en az 4 eğitim, 5 galeri görüntüsü). Bu şartı 75 sınıf (1.956 görüntü) sağlıyor; kalan 26 sınıfta yalnızca 7-12 kullanılabilir görüntü olduğu için (toplam 264 görüntü) ayrık kümeler kurulamıyor ve bu sınıflar birincil değerlendirmenin dışında kaldı. Dışarıda kalan sınıflar `docs/CLASS_COVERAGE_AUDIT.md` dosyasında tek tek listelenmiştir. Dolayısıyla bu raporda verilen yaklaşık 0,80 Hit@1 değeri **75 sınıflı, kapalı kümeli bir protokolün sonucudur; 101 sınıfın tamamı üzerinde bir değerlendirme değildir.**'));
kids.push(...FIG('class_hist', 430, 'Şekil 2. Sınıf başına kullanılabilir görüntü sayısı. Mavi çubuklar birincil kümedeki 75 sınıfı, gri çubuklar kapsam dışı kalan 26 sınıfı göstermektedir.'));
kids.push(P('Kaynak veride kalite sorunları da bulduk. SHA-256 ve algısal hash (pHash ≤ 12 ve dHash ≤ 18, eşikler tüm aday çiftler gözle incelenerek seçildi) kuralıyla 26 yinelenen grup bulundu. Bunların 16\'sında (32 görüntü) aynı fotoğraf farklı etiketlerle kayıtlıydı (ör. limon ile portakal antraknozu); gerçek etiket bilinemediği için hepsi çıkarıldı. Aynı sınıf içindeki 11 kopya da çıkarıldı; toplam 43 görüntü elendi ve 2.220 görüntü kaldı.'));
kids.push(H2('2.2 PlantSeg v2 (harici veri seti)'));
kids.push(P('PlantSeg, gerçek ortamda çekilmiş bitki hastalığı görüntülerinden oluşan ve segmentasyon için hazırlanmış büyük bir veri setidir [2] (Zenodo 13958858, CC BY-NC-ND 4.0 [8]; ham görüntüleri paylaşmıyoruz). Yerel olarak çıkardığımız sürümde 11.458 görüntü bulunuyor. **Sınıf sayısında bir tutarsızlık var:** yayınlanan arXiv sürümü 115 hastalık kategorisi ve 34 bitki türü bildiriyor [2]; bizim Metadatav2.csv dosyamızda 114 farklı hastalık adı ve 35 bitki etiketi, coco_annotations.json dosyasında ise 117 kategori ve 11.480 görüntü kaydı var (üç kategori yalnızca coco dosyasında geçiyor). Bu farkın nedenini (sürüm farkı ya da etiketleme düzeltmesi) doğrulayamadık; dergi sürümüne erişemediğimiz için oradaki sayıları kullanmadık. Bu çalışma için bu fark sonucu etkilemiyor, çünkü yalnızca bitki ve hastalık adı SegPPD-101 ile birebir örtüşen sınıfları kullandık.'));
kids.push(P('Çapraz veri seti sızıntı taramasında SegPPD-101\'in 2.263 görüntüsünün tamamını PlantSeg görüntüleriyle karşılaştırdık (256×256\'ya yeniden boyutlandırılmış görüntüler üzerinde hash kuralı ve donmuş DINOv2-B kosinüs benzerliği ≥ 0,85). **867 PlantSeg görüntüsü (%7,6) SegPPD-101\'deki bir görüntünün kopyası ya da kırpımı çıktı** (229\'u hash kuralıyla, 862\'si gömme benzerliğiyle); yani iki veri seti ortak web kaynaklarından beslenmiş. Bu görüntülerin hepsi çıkarıldı. Eşik değeri, 0,92\'lik daha sıkı bir değerin kırpılmış kopyaları kaçırdığını gözle görerek 0,85 olarak seçildi. Sıkı eşleştirilen 11 sınıfta 1.621 görüntüden 136\'sı elenince **1.485 harici sorgu** kaldı; arama, 75 sınıflı SegPPD-101 galerisinde yapıldı. Eşleme yalnızca bitki ve hastalık adı birebir aynı olan sınıfları içeriyor (elma uyuz ve pas, muz cordana yaprak lekesi, fasulye pası, kahve pası, salatalık külleme, üzüm kara çürüklük, çilek yaprak yanıklığı, domates bakteriyel leke, yaprak küfü ve geç yanıklık). PlantSeg\'de sağlıklı sınıf yok. Ayrıca görüntüler orijinal çözünürlükte web fotoğraflarıdır ve 256×256\'ya sıkıştırılarak kullanılmıştır; bu bir alan kaymasıdır (Bölüm 7).'));

// ================= 3
kids.push(H1('3. İlgili Çalışmalar'));
kids.push(P('Tablo 1, doğrulayabildiğimiz çalışmaları özetlemektedir (ayrıntılı doğrulama kaydı `docs/LITERATURE_COMPARISON.md` dosyasındadır). Kaynak sayfaları otomatik özetleme ile okuduk; resmî sunumdan önce sayıların PDF ile tekrar kontrol edilmesini öneririz. TaxoSeg çalışmasına (ScienceDirect) erişemedik (HTTP 403) ve başka bir kaynakta bulamadık; bu nedenle tabloya ve karşılaştırmaya almadık. Shuffle-PG için yayıncı sayfası da erişime kapalıydı, yalnızca özet düzeyindeki bilgiyi kullandık.'));
kids.push(CAPT('Tablo 1. Doğrulanan ilgili çalışmalar. Hiçbirinin ölçütü bizim Hit@1 değerimizle doğrudan karşılaştırılabilir değildir.'));
kids.push(TABLE([1250, 1000, 1900, 1850, 2000, 1300], ['Çalışma', 'Görev', 'Veri (sınıf)', 'Yöntem', 'Ölçüt ve sonuç', 'Karşılaştırılabilir mi?'], [
  ['Ge ve ark. 2025 (SegPPD-FS) [1]', 'Az örnekli segmentasyon', 'SegPPD-101 (101 sınıf; 80 temel + 21 yeni)', 'ResNet-50, HDMNet + SFEM + HPKIM', 'mIoU %71,19 (1-shot) - %72,74 (8-shot)', 'Hayır: aynı veri, farklı görev (segmentasyon)'],
  ['Wei ve ark. 2024/2025 (PlantSeg) [2]', 'Segmentasyon veri seti', '11.458 görüntü, 115 kategori, 34 bitki (arXiv)', 'DeepLabv3/3+, SAN, SegNeXt', 'mIoU en iyi %44,52 (tablo)', 'Hayır: segmentasyon'],
  ['Pan ve ark. 2025 (ChatLeafDisease) [3]', 'Sıfır örnekli sınıflandırma', 'PlantVillage domates (6 sınıf); PlantSeg alt kümesi', 'GPT-4o + düşünce zinciri istemi', 'Doğruluk %88,9 (PlantVillage); %77,3 (PlantSeg)', 'Hayır: sınıflandırma, farklı sınıflar'],
  ['Jin ve ark. 2025 (Shuffle-PG) [4]', 'CBIR', 'Kendi topladıkları hastalık ve zararlı setleri (sınıf sayısı doğrulanamadı)', 'ShuffleNet v2 + grup evrişimi, kontrastif kayıp', 'mAP %97,7 (hastalık), %98,8 (zararlı)', 'Hayır: farklı veri ve protokol'],
  ['Yang ve ark. 2024 (hashing) [5]', 'Erişim (hashing)', 'PlantVillage (38 sınıf); PlantDoc; Plant Disease (12 bitki)', 'VGG-11 + 64 bit hash katmanı', 'MAP 0,9974 (PlantVillage); 0,8204 (PlantDoc)', 'Hayır: farklı veri; PlantDoc zorluk olarak en yakın'],
  ['Peng ve Wang 2022 [6]', 'Erişimle tanıma', 'PlantVillage, kahve (2.209), narenciye (609)', 'YOLOv5 + ResNet-50, softmax + triplet, kosinüs arama', 'Top-1 %97,84 (kapalı küme); %89,09 ve %91,67 (açık küme)', 'Hayır: farklı veri ve protokol']], { size: 15, leftCols: 1 }));
kids.push(SP());
kids.push(P('Tablo 1\'deki ilk üç çalışma segmentasyon ya da sınıflandırma yaptığı için ölçütleri (mIoU, doğruluk) bir erişim ölçütü olan Hit@1 ile aynı sıraya konamaz; aynı SegPPD-101 görüntüleri üzerinde bile bu karşılaştırma geçerli olmaz. Son üç çalışma erişim üzerinedir ve bize kavramsal olarak en yakın olanlardır; fakat laboratuvar görüntüleri (PlantVillage) ya da kendi topladıkları setlerle çalıştıkları ve farklı sınıf sayısı, bölünme ve ölçüt tanımı kullandıkları için doğrudan karşılaştırılamazlar. PlantVillage\'daki 0,99 civarı değerler tek yaprak, sade arka plan gibi kolay bir ortamı yansıtır; gerçek ortam zorluğuna en yakın sonuç PlantDoc üzerindeki 0,820 MAP değeridir, ancak onun da protokolü bizimkinden farklıdır. Bildiğimiz kadarıyla aynı SegPPD-101 erişim protokolünde yayınlanmış bir sonuç bulamadık; bu nedenle **hiçbir "en iyi sonuç" iddiasında bulunmuyoruz.** Bizim yaklaşımımız bu çalışmalardan, öznitelik uzayını sınıf etiketleriyle (denetimli kontrastif öğrenme) öğrenip arama adımını tam Öklid olarak sabit tutması ve sızıntısız, birden fazla kilitli test kümesiyle (iç ve harici) değerlendirilmesi bakımından ayrılır.'));

// ================= 4
kids.push(H1('4. Yöntemler ve Mimari'));
kids.push(P('Yöntemleri deneme sırasına göre anlatıyoruz. Bu bölümde yalnızca ne yaptığımızı açıklıyoruz; ölçülen sonuçlar Bölüm 6\'dadır. Çoğu ön deneyi, son kararlar için kullanmadığımız bir geliştirme (dev) kümesinde yaptık.'));
kids.push(H2('4.1 Donmuş ResNet50 (temel yöntem)'));
kids.push(P('Hocamızın yöntemi: ImageNet-1K V2 ağırlıklı torchvision ResNet50, hiç eğitilmeden kullanıldı. Tek bir ileri geçişle iki öznitelik çıkarıldı: son tam bağlı katmandan gelen 1000 boyutlu ham logit (birincil) ve global ortalama havuzlama sonrası 2048 boyutlu vektör (karşılaştırma). Arama tam Öklid mesafesiyle yapıldı; eşitlikte görüntü kimliği kullanıldı. Şekil 3\'te görüldüğü gibi galeri vektörleri çevrimdışı hesaplanıyor. Normalleştirmenin etkisini de denedik (L2 normalleştirme, Bölüm 6).'));
kids.push(...FIG('arch_baseline', 600, 'Şekil 3 (Şekil A). Temel CBIR mimarisi: görüntü, donmuş ResNet50, öznitelik vektörü (2048-B ve 1000-B), tam Öklid mesafe ve sıralı Top-K sonuçlar.'));
kids.push(H2('4.2 Çok katmanlı öznitelikler, el yapımı öznitelikler ve füzyon'));
kids.push(P('Eğitimsiz yöntemi iyileştirmek için önce ResNet50\'nin layer3 ve layer4 çıktılarını farklı havuzlamalarla (ortalama, GeM, maksimum) ve iki ölçekte (256 ve 320 piksel) birleştirdik, sorgu genişletmesi ekledik. Ardından klasik el yapımı öznitelikleri denedik: HSV renk histogramı, Lab renk momentleri, LBP, HOG, GLCM, Gabor ve kenar öznitelikleri. Son olarak bu klasik özniteliklerden görüntü haritaları (LBP, Gabor, Lab, HSV, gradyan, bitki indeksi ve CLAHE) üretip bunları ResNet50\'ye girdi olarak verdik; haritaları RGB özniteliğine birleştirme (concat) ve dikkat (attention) füzyonuyla bağladık. Bu deneylerin hiçbiri belirgin bir kazanç sağlamadı (Bölüm 6.2); bu olumsuz bulguyu raporda bilerek tutuyoruz.'));
kids.push(H2('4.3 ResNet50 ince ayar ve ArcFace'));
kids.push(P('ResNet50\'nin layer3 ve layer4 blokları (22,6 milyon parametre), 256 boyutlu L2 normalize bir gömme başlığı ve ArcFace kaybıyla (ölçek 30, marjin 0,3) eğitildi. Bu, Protokol B\'de "mevcut eğitimli sistemimiz" olarak kullanıldı.'));
kids.push(H2('4.4 Donmuş DINOv2 ve metrik öğrenme'));
kids.push(P('Gövdeyi DINOv2 [10] ile değiştirdik (timm vit_small ve vit_base_patch14_dinov2, LVD-142M ön eğitimi). Önce hiç eğitmeden CLS, yama ortalaması ve GeM özniteliklerini 224 ve 336 pikselde denedik; en iyisi 224 pikselde CLS oldu. Sonra DINOv2-S\'te yalnızca başlığı, ardından son 4 bloğu da eğittik ve üç kayıp fonksiyonunu karşılaştırdık: ArcFace [12], Proxy-Anchor [13] ve denetimli kontrastif kayıp (SupCon) [11]. Kayıp seçimi geliştirme kümesinde mAP\'e göre yapıldı.'));
kids.push(H2('4.5 Önerilen nihai mimari: DINOv2-B + SupCon'));
kids.push(P('Şekil 4\'te gösterilen önerilen model, gerçek kodumuzda şu parçalardan oluşur: DINOv2-B (ViT-B/14) gövdesinin ilk 8 bloğu donuk, son 4 bloğu ve çıkış normalizasyonu eğitilebilir (28,6 milyon eğitilebilir parametre; blok başına öğrenme oranı 3e-5, her önceki blokta 0,8 ile çarpılarak azaltılır); CLS jetonunun 768 boyutlu çıktısı dropout (0,2) ve doğrusal katmanla 256 boyutlu, L2 normalize bir gömmeye dönüştürülür (başlık için öğrenme oranı 1e-3). Eğitim yalnızca train_B görüntüleriyle (772 görüntü), 16 sınıf × 4 görüntülük sınıf dengeli batch\'lerle, kırpma, yatay/dikey çevirme ve renk oynatma artırmalarıyla, AdamW ve kosinüs çizelgesiyle yapıldı; geliştirme kümesinde mAP\'e göre erken durdurma uygulandı (sabır 10, en iyi epoch ağırlıkları geri yüklendi). Kayıp, denetimli kontrastif kayıptır (sıcaklık 0,1): aynı sınıftaki görüntüler yaklaştırılır, farklı sınıftakiler uzaklaştırılır. Şekil 4\'te kırmızı etiketli parçalar yalnızca eğitimde, yeşil etiketli parçalar çıkarımda bulunur. Çıkarımda görüntü ve yatay çevrilmişi aynı ağdan geçirilir, iki gömme toplanıp L2 normalleştirilir ve arama yine tam Öklid mesafesiyle yapılır; yani erişim adımı temel yöntemle aynıdır.'));
kids.push(...FIG('arch_final', 610, 'Şekil 4 (Şekil B). Önerilen nihai mimari: (a) yalnızca eğitimde kullanılan parçalar (DINOv2-B, son 4 blok, başlık, SupCon kaybı), (b) çıkarımda kullanılan parçalar (tam Öklid arama). Üç tohumlu topluluk isteğe bağlı ve ikincil bir sonuçtur.'));
kids.push(P('**Üç tohumlu topluluk.** Aynı ayarla üç ayrı tohumla (0, 1, 2) eğitilen modellerin 256 boyutlu gömmeleri √(1/3) ile ağırlıklandırılıp art arda birleştirilir (768 boyut); böylece kare Öklid mesafesi üç modelin kare mesafelerinin ortalamasına eşit olur ve arama hâlâ tam Öklid kalır. Önceden ilan ettiğimiz başlık sonuç, topluluk değil, üç tekil modelin ortalamasıdır; topluluk ve füzyonlar ikincil sonuç olarak raporlandı. Maske kırpma (oracle ROI), topluluk ve füzyon gibi ek denemeler nihai modele alınmadı.'));

// ================= 5
kids.push(H1('5. Değerlendirme Protokolü ve Ölçütler'));
kids.push(P('İlgililik, aynı veri seti sınıf etiketine sahip olmaktır (etiket uyumu); bu tarımsal olarak doğrulanmış bir teşhis değildir. Etiketler yalnızca sonucu puanlamak için kullanılır, adayları filtrelemek için kullanılmaz. Tablo 2 protokolleri ve sayılarını göstermektedir.'));
kids.push(CAPT('Tablo 2. Değerlendirme protokolleri ve kümelerin boyutları.'));
kids.push(TABLE([1500, 1450, 1350, 1100, 1500, 2400], ['Protokol', 'Sınıf', 'Eğitim', 'Galeri', 'Sorgu', 'Not'], [
  ['A (eğitimsiz temel)', '75', '-', '1.731', '225 test; 150 dev (dev galeri 1.581)', 'Hocanın yöntemi; eğitimli sistemler galeriyle eğitildi (keşifsel); ikincil: 26 sınıf × 1 sorgu'],
  ['B (sızıntısız)', '75', '772', '809', '225 kilitli test; 150 dev', 'Eğitim/galeri/sorgu görüntü ve grup düzeyinde ayrık'],
  ['C', '75', '772 (B ile aynı)', '667', '142 kilitli test', 'B galerisinden ayrılmış (1-2 sorgu/sınıf)'],
  ['B + C', '75', '-', '809 / 667', '367', 'Sorgular birleştirilir; galeriler farklı olduğundan yaklaşık'],
  ['Harici (PlantSeg)', '11 (arama 75 sınıf)', 'B ile aynı model', '809', '1.485', 'Sızıntı taramasından geçen görüntüler']], { size: 15, leftCols: 1 }));
kids.push(SP());
kids.push(P('Her sistemde arama tam Öklid mesafesiyle yapıldı. **Hit@K**, ilk K sonuçta en az bir ilgili görüntü olup olmadığıdır; **Hit@1 = P@1** olup 1-en yakın komşu doğruluğuna eşittir. **Precision@K** ilk K sonucun ilgili olanlarının oranı, **Recall@K** ise bu ilgili görüntülerin sorgunun tüm ilgili galeri görüntülerine oranıdır. **AP**, sorgu için tüm sıralama üzerinden hesaplanan ortalama kesinliktir ve **mAP** bunun sorgular üzerinden ortalamasıdır. Başlık sayılar **sınıf ortalamasıdır** (makro: önce sınıf içinde, sonra sınıflar arasında ortalama); sorgu ağırlıklı (mikro) ortalama, B ve A\'da sınıf başına 3 sorgu olduğu için aynıdır. %95 güven aralıkları sınıflar üzerinden bootstrap ile (2.000 tekrar) hesaplandı; eğitim tohumu değişkenliğini ise üç tohumun ayrı değerleriyle gösterdik.'));
kids.push(P('**Kesinlik-anma (P-R) eğrisi:** her sorgu için tüm galeri sıralaması üzerinde her sırada kesinlik ve anma hesaplandı, kesinlik 11 anma seviyesinde (0, 0,1, ..., 1) enterpole edildi (o seviyenin üstündeki en yüksek kesinlik), sonra önce sınıf içinde, sonra sınıflar arasında ortalandı. **ROC-AUC:** her sorgu-galeri çifti bir örnek, aynı sınıftan olma durumu pozitif etiket, negatif kare Öklid mesafesi skor olarak alınıp tüm çiftler üzerinde (sorgu sayısı × galeri) tek bir ROC hesaplandı; pozitifler negatiflerden çok az olduğu için AUC değerleri yüksek görünür. **F1:** gerçek bir sınıflandırıcı eğitmedik; sorgunun ilk sonucunun etiketini tahmin sayıp 75 sınıf üzerinde makro F1 hesapladık; bu nedenle doğruluk Hit@1 ile aynıdır ve F1 ek bilgi değil, aynı kararın başka bir özetidir. Oylama (kNN-vote) doğruluğunu Hit@1 yerine kullanmadık.'));

// ================= 6
kids.push(H1('6. Deneysel Sonuçlar ve Ablasyon Çalışması'));
kids.push(P('Sonuçları farklı protokolleri birbirine karıştırmadan, aşamalı olarak sunuyoruz: önce Protokol A\'daki eğitimsiz temel yöntem, sonra geliştirme kümesi ön denemeleri, sonra Protokol B kilitli testi, en sonda C ve harici testler.'));
kids.push(H2('6.1 Eğitimsiz temel yöntem (Protokol A)'));
kids.push(P('Donmuş ResNet50 ile 1000 boyutlu logit Hit@1 0,293 [0,227; 0,360], P@10 0,170 ve mAP 0,118 verdi; 2048 boyutlu vektör Hit@1 0,391 [0,316; 0,458], P@10 0,212 ve mAP 0,145 verdi (225 sorgu, galeri 1.731; ana bölünme, tohum 42). Rastgele seçimde P@K yaklaşık 0,013\'tür. 2048 boyutlu vektör logitten her K değerinde daha iyidir. L2 normalleştirme tutarlı bir fark yaratmadı (logit 0,262, 2048-B 0,373; güven aralıkları örtüşüyor). Aynı bölünmede çok katmanlı ve çok ölçekli öznitelik ile sorgu genişletmesi (keşifsel, A test sorguları daha önce görüldüğü için) Hit@1 0,453, P@10 0,260 verdi.'));
kids.push(H2('6.2 Olumsuz bulgular: el yapımı öznitelikler, görüntü haritaları, dikkat ve ROI'));
kids.push(P('Tablo 3, geliştirme kümesindeki (150 sorgu) bu denemeleri göstermektedir. El yapımı öznitelik vektörlerinin hiçbiri derin öznitelikle yarışamadı: en iyisi HSV, mAP 0,050 (derin öznitelik 0,154). Haritaların ResNet50\'ye girdi olarak verilmesi tek başına en fazla mAP 0,160 (CLAHE) verdi, bu da RGB\'nin 0,154\'üne eşdeğerdir. Eğitilen bir başlıkla RGB girdisine harita eklemek (birleştirme 0,516-0,557, dikkat 0,475-0,524 mAP) yalnızca RGB başlığın 0,548 mAP değerini düşürdü ya da gürültü düzeyinde değiştirdi. El yapımı vektörleri derin özniteliklerle birleştirmek de zarar verdi (P@1 0,493 iken ağırlık 0,25\'te 0,487, ağırlık 1\'de 0,447). Hedef bölgeyi maske kutusuyla kırpmak (oracle ROI, gerçek maske gerektirdiği için gerçek kullanımda mümkün değil) donmuş ResNet50\'de P@10 değerini yalnızca +0,016 [-0,004; 0,034] değiştirdi. Yani sorunun nedeni arka plan seyrelmesi gibi görünmüyor, özniteliklerin kendisi yetersiz.'));
kids.push(CAPT('Tablo 3. Geliştirme kümesinde (150 sorgu) olumsuz bulgular (mAP; parantez içinde P@1).'));
kids.push(TABLE([3600, 1500, 1500, 1700], ['Deneme', 'Tek başına', 'RGB + harita (concat)', 'RGB + harita (dikkat)'], [
  ['RGB girdili eğitilmiş başlık (referans)', '-', '0,548 (0,603)', '-'],
  ['HSV (vektör: 0,050 (0,167); harita)', '0,097 (0,307)', '0,556 (0,620)', '0,518 (0,543)'],
  ['Lab (vektör momentleri: 0,028 (0,040); harita)', '0,123 (0,347)', '0,557 (0,613)', '0,508 (0,540)'],
  ['LBP (vektör: 0,033 (0,047); harita)', '0,059 (0,140)', '0,520 (0,580)', '0,492 (0,503)'],
  ['Gabor (vektör: 0,022 (0,013); harita)', '0,041 (0,080)', '0,525 (0,580)', '0,475 (0,517)'],
  ['CLAHE (harita)', '0,160 (0,387)', '0,547 (0,613)', '0,524 (0,560)'],
  ['HOG, GLCM, kenar (yalnızca vektör)', '0,025 / 0,024 / 0,024', '-', '-']], { size: 16, leftCols: 1 }));
kids.push(SP());
kids.push(H2('6.3 Protokol B kilitli test: ResNet50\'den DINOv2-B\'ye'));
kids.push(P('Tablo 4 ve Şekil 5, Protokol B kilitli test kümesindeki (225 hiç kullanılmamış sorgu, 75 sınıf, galeri 809) ablasyon adımlarını göstermektedir. Aday listesi ve başlık tanımı test koşulmadan önce dondurulmuş, test bir kez çalıştırılmıştır. Donmuş ResNet50 Protokol B\'de Hit@1 0,262 (logit) ve 0,320 (2048-B) vermiştir; ResNet50\'yi ArcFace ile ince ayarlamak 0,511\'e çıkarmıştır. Gövdeyi DINOv2\'ye çevirmek, hiç eğitmeden bile 0,538 (S) ve 0,627 (B) vermiş; son 4 bloğun SupCon ile eğitilmesi DINOv2-B\'de 0,807 (tohumlar 0,813; 0,796; 0,813) ve mAP 0,742 vermiştir. Üç tohumun topluluğu 0,822 [0,769; 0,876], ek olarak ResNet50 ile füzyon 0,840 vermiştir, ancak bu kazançlar anlamlı değildir (füzyon için +0,018, %95 GA [-0,004; 0,044]).'));
kids.push(CAPT('Tablo 4. Ablasyon (Protokol B kilitli test; 225 sorgu, 75 sınıf, galeri 809). Doğruluk = 1-en yakın komşu doğruluğu = Hit@1; F1 makro; ROC-AUC sorgu-galeri çiftleri üzerinden. 3 tohum ort. satırları tohum ortalamasıdır.'));
kids.push(TABLE([3700, 1000, 850, 900, 800, 800, 950], ['Sistem', 'Doğruluk (Hit@1)', 'P@10', 'Hit@10', 'mAP', 'F1', 'ROC-AUC'], tB, { boldRows: [8] }));
kids.push(SP());
kids.push(...FIG('hit_map', 520, 'Şekil 5. Ablasyon adımlarında Hit@1 ve mAP değerleri (Protokol B kilitli test). Kırmızı çizgi 0,80 hedefidir; çubuk uçları Hit@1 için %95 sınıf-bootstrap güven aralığıdır.'));
kids.push(P('Geliştirme kümesinde (Tablo 5) önemli adımlar şöyleydi: DINOv2-S\'te yalnızca başlığı eğitmek mAP 0,514, son 4 bloğu da açmak 0,597 verdi; kayıplarda SupCon (0,663) > Proxy-Anchor (0,636) > ArcFace (0,597) çıktı; DINOv2-B ise mAP 0,750 ve P@1 0,815 verdi. **Kilitli testte ise sıralama Hit@1 açısından farklıdır:** DINOv2-S\'te ArcFace (0,760) ve Proxy-Anchor (0,748), SupCon\'dan (0,717) daha iyidir; SupCon yalnızca P@10 (0,591) ve mAP\'te (0,658) öndedir. Kayıp seçimini geliştirme kümesinde mAP\'e göre yaptığımız için SupCon\'u seçtik; DINOv2-B\'de ArcFace ve Proxy-Anchor denenmedi, dolayısıyla adil bir kayıp karşılaştırması DINOv2-B için yapılmamıştır.'));
kids.push(CAPT('Tablo 5. Geliştirme kümesinde (150 sorgu) ince ayar kapsamı ve kayıp ablasyonu; model seçimi bu kümede yapılmıştır.'));
kids.push(TABLE([5200, 1300, 1300, 1300], ['Sistem', 'mAP', 'P@1', 'P@10'], [
  ['DINOv2-S, yalnızca başlık + ArcFace', '0,514', '0,667', '0,464'], ['DINOv2-S, son 4 blok + ArcFace', '0,597', '0,720', '0,546'],
  ['DINOv2-S, son 4 blok + Proxy-Anchor (3 tohum ort.)', '0,636', '0,724', '0,572'], ['DINOv2-S, son 4 blok + SupCon (3 tohum ort.)', '0,663', '0,725', '0,596'],
  ['DINOv2-B, son 4 blok + SupCon (3 tohum ort.)', '0,750', '0,815', '0,667'], ['(karşılaştırma) ResNet50 ince ayar + ArcFace', '0,420', '0,527', '0,399']], { boldRows: [4] }));
kids.push(SP());
kids.push(P('Şekil 6\'da görüldüğü gibi P@K, K büyüdükçe düşer; çünkü sınıf başına ortalama yaklaşık 10 ilgili görüntü vardır ve P@K için üst sınır min(R,K)/K\'dır. Önerilen model P@10\'da 0,651 ile bu sınırın büyük bölümüne ulaşmıştır. Şekil 7a\'daki kesinlik-anma eğrisi, önerilen modelin anma değerinin her düzeyinde diğer sistemlerin üstünde olduğunu gösterir (anma 0\'da yaklaşık 0,90, anma 1\'de yaklaşık 0,55; donmuş ResNet50\'de 0,48 ve 0,05). Şekil 7b ve 7c\'deki ROC eğrilerinde AUC önerilen modelde 0,989, ince ayarlı ResNet50\'de 0,914, donmuş ResNet50\'de 0,765\'tir; yakınlaştırılmış panelde (7c) fark çok daha belirgindir: yanlış pozitif oranı 0,02 iken önerilen model gerçek pozitiflerin yaklaşık 0,92\'sini, ince ayarlı ResNet50 yaklaşık 0,55\'ini yakalar.'));
kids.push(...FIG('prk_rk', 540, 'Şekil 6. K değerine göre Precision@K ve Recall@K (Protokol B kilitli test, sınıf ortalaması; kesikli çizgi P@K için üst sınırdır).'));
kids.push(...FIG('pr_roc', 560, 'Şekil 7. (a) Kesinlik-anma eğrisi (11 noktalı enterpole), (b) ROC eğrisi ve (c) yakınlaştırılmış ROC (Protokol B kilitli test; ilgililik = aynı sınıf etiketi; ROC tüm sorgu-galeri çiftleri üzerinden).'));

// ================= 7
kids.push(H1('7. Nihai Mimari ve Sonuçlar'));
kids.push(P('Önerdiğimiz model Şekil 4\'teki DINOv2-B + SupCon mimarisidir. Tablo 6 bu modelin tüm test kümelerindeki sonuçlarını (3 tohum ortalaması) birlikte vermektedir; temel yöntemle karşılaştırma için Tablo 7 aynı kümelerde donmuş ResNet50\'yi ve donmuş DINOv2-B\'yi göstermektedir. Önerilen model, hocamızın zorunlu ResNet50 temelinin (Hit@1 0,320 / 0,333 / 0,149) her kümede çok üzerindedir.'));
kids.push(CAPT('Tablo 6. Önerilen modelin tüm test kümelerindeki sonuçları. [ ] içinde %95 sınıf-bootstrap güven aralığı; "-" hesaplanmamıştır.'));
kids.push(TABLE([2300, 650, 1000, 1550, 650, 750, 650, 600, 800], ['Test kümesi', 'Sorgu', 'Sınıf', 'Hit@1 [GA]', 'P@10', 'Hit@10', 'mAP', 'F1', 'ROC-AUC'], tFinal, { size: 15 }));
kids.push(SP());
kids.push(CAPT('Tablo 7. Temel yöntemle karşılaştırma: Hit@1 (aynı protokol içinde).'));
kids.push(TABLE([4200, 1700, 1700, 1700], ['Sistem', 'B (225 sorgu)', 'C (142 sorgu)', 'Harici (1.485; 11 sınıf)'], [
  ['ResNet50 donmuş, 2048-B (temel)', '0,320', '0,333', '0,149'], ['DINOv2-B donmuş', '0,627', '0,567', '0,484'],
  ['DINOv2-B son 4 blok + SupCon (önerilen)', '0,807', '0,767', '0,608']], { boldRows: [2] }));
kids.push(SP());
kids.push(P('Şekil 8\'de görüldüğü gibi Protokol B\'de önerilen model 75 sınıfın yalnızca birinde hiç doğru sonuç bulamamıştır (tohum ortalaması yuvarlanarak 0/3), 40 sınıfta üç sorgunun üçünü de bulmuştur; donmuş ResNet50\'de 29 sınıf 0/3, yalnızca 4 sınıf 3/3\'tür. En zayıf sınıflar limon sağlıklı (0,11), pamuk fusarium solgunluğu (0,22) ve hops besin eksikliği, kassava bakteriyel yanıklık, limon isli küf ve domates küf (0,33) gibi birbirine ya da sağlıklı yaprağa görsel olarak çok benzeyen sınıflardır. Şekil 9, rastgele (sabit tohumla, el ile seçilmeden) seçilmiş iki başarılı ve iki başarısız sorgu için temel yöntemin ve önerilen modelin ilk 4 sonucunu göstermektedir: önerilen model aynı sınıftan, farklı kadrajlardaki görüntüleri getirirken, temel yöntem çoğunlukla rengi ya da genel görünümü benzer ama farklı sınıftan görüntüler getirmiştir. Başarısız örneklerde ise hatalar benzer hastalık görünümlerinden (ör. domates küfü ile geç yanıklık, fasulye pası ile açısal yaprak lekesi) kaynaklanmaktadır.'));
kids.push(...FIG('per_class', 540, 'Şekil 8. Sınıf başına erişim başarısı (Protokol B kilitli test). Solda sınıfın 3 sorgusundan kaçının doğru bulunduğu (önerilen modelde 3 tohum ortalamasının en yakın tam sayıya yuvarlanmış hâli), sağda önerilen modelin en zayıf 12 sınıfı.'));
kids.push(...FIG('retrieval_examples', 610, 'Şekil 9. Rastgele seçilmiş iki başarılı ve iki başarısız sorgu için ilk 4 sonuç (Protokol B kilitli test; her sorguda üst satır donmuş ResNet50, alt satır DINOv2-B + SupCon tohum 0; yeşil çerçeve = doğru sınıf, kırmızı = yanlış sınıf).'));
kids.push(P('**Dahili ve harici sonuçlar neden farklı?** Şekil 10\'da görüldüğü gibi önerilen modelin Hit@1 değeri B\'de 0,807, C\'de 0,767 ve harici sette 0,608\'dir; sıralama (ResNet50 < donmuş DINOv2-B < önerilen model) her üç kümede de korunmuştur. Aynı 11 sınıfta SegPPD-101 içindeki değer 0,778 iken harici değer 0,608\'dir, yani yaklaşık 17 puanlık düşüş vardır. Olası nedenler: (i) alan kayması, çünkü PlantSeg görüntüleri orijinal çözünürlüklü ve farklı kadrajlı web fotoğraflarıdır, biz bunları 256×256\'ya sıkıştırdık; (ii) benzerliği yüksek kopya görüntüler çıkarıldığı için kalan sorgular daha zordur; (iii) etiket eşlemesi yalnızca ad eşleşmesine dayanır, iki veri setinde aynı adlı sınıfların görsel kapsamı farklı olabilir; (iv) harici test yalnızca 11 sınıfı kapsar ve sınıf bazında farklar büyüktür (üzüm kara çürüklüğü SegPPD-101 içinde 1,00 iken harici sette 0,31). Bu nedenlerin payını ayıramıyoruz. Güven aralığı da geniştir (sınıf bazında [0,466; 0,760]).'));
kids.push(...FIG('generalisation', 540, 'Şekil 10. Üç sistemin SegPPD-101 kilitli testleri B ve C ile harici PlantSeg kümesindeki Hit@1 değerleri (%95 sınıf-bootstrap güven aralığıyla).'));
kids.push(P('**0,80 hedefi hakkında:** Protokol B\'de nokta tahmini 0,807 ile 0,80\'in üzerindedir; fakat güven aralığı [0,756; 0,856] 0,80\'in altını da içerir ve tohumlardan biri 0,796\'dır. İkinci kilitli kümede değer 0,767, birleşik 367 sorguda 0,794 [0,741; 0,844]\'tür. Dolayısıyla 0,80\'in istatistiksel olarak kanıtlandığını söyleyemeyiz; doğru ifade, 75 sınıflı SegPPD-101 protokolünde Hit@1\'in yaklaşık 0,79-0,81 olduğu, harici ve alan dışı görüntülerde ise yaklaşık 0,61\'e düştüğüdür. Maliyet olarak model 28,6 milyon eğitilebilir parametreye sahiptir; eğitim bir RTX 4050 (6 GB) üzerinde yaklaşık 10-15 dakika sürmüş ve yaklaşık 3 GB bellek kullanmıştır; 809 görüntülük galeride 256 boyutlu vektörlerle tam arama sorgu başına medyan 0,26 ms sürer.'));

// ================= 8
kids.push(H1('8. Sonuç ve Sınırlılıklar'));
kids.push(P('Hocamızın donmuş ResNet50 yöntemi SegPPD-101 üzerinde Hit@1 bakımından yaklaşık 0,26-0,39 düzeyinde kalmıştır. Erişim adımını (tam Öklid arama) değiştirmeden yalnızca öznitelik uzayını DINOv2-B ve denetimli kontrastif öğrenmeyle öğrenerek, 75 sınıflı sızıntısız protokolde bu değeri yaklaşık 0,79-0,81\'e çıkardık. Ablasyon, kazancın büyük kısmının gövdenin DINOv2\'ye geçmesinden ve son blokların etiketlerle ince ayarlanmasından geldiğini, el yapımı öznitelikler, dikkat füzyonu, ROI kırpma ve topluluk/füzyonun ise anlamlı bir katkı sağlamadığını göstermiştir.'));
kids.push(P('Sınırlılıklar şunlardır. (i) **Kapsam:** değerlendirme 101 sınıfın 75\'ini kapsar; 26 sınıfta yalnızca 7-12 görüntü olduğu için ayrık kümeler kurulamamıştır ve bu sınıfları kapsamayan bir sonucu 101 sınıfa genellemek doğru olmaz. (ii) **Küçük sınıflar ve belirsizlik:** sınıf başına 1-3 test sorgusu vardır, güven aralıkları geniştir ve 0,80 hedefi kesinleştirilememiştir; iki kilitli küme de artık kullanılmıştır, yeni model seçimi için yeni bir kilitli küme gerekir. (iii) **Etiket gürültüsü:** kaynak veride aynı fotoğrafın farklı etiketle yer aldığı 16 grup bulunmuştur; tespit edilemeyen hatalar kalmış olabilir. (iv) **Kaynak çakışması:** PlantSeg ile SegPPD-101 ortak kaynaklı çıkmıştır; taramada yakalanamayan, daha fazla düzenlenmiş kopyalar kalmış olabilir. (v) **Harici genelleme:** başarı harici sette 0,61\'e düşmüştür ve bu test yalnızca 11 sınıfı kapsar. (vi) **İlgililik tanımı:** aynı sınıf etiketi, onaylanmış bir tarımsal teşhis anlamına gelmez; sistem bir tanı aracı değildir. (vii) **Seçim etkisi:** kayıp fonksiyonu geliştirme kümesinde mAP\'e göre seçilmiştir ve DINOv2-S\'te SupCon Hit@1\'de ArcFace\'ten zayıf çıkmıştır.'));

kids.push(H1('Kaynaklar'));
const refs = [
  'Ge, Z., Fan, X., Zhang, J., Jin, S. SegPPD-FS: Segmenting plant pests and diseases in the wild using few-shot learning. Plant Phenomics, 7(4):100121, 2025. doi:10.1016/j.plaphe.2025.100121. https://pmc.ncbi.nlm.nih.gov/articles/PMC13109344/',
  'Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., Huang, Z. PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation. arXiv:2409.04038, 2024 (dergi sürümü: https://www.nature.com/articles/s41597-025-06513-4; erişilemedi).',
  'Pan, J., Zhong, R., Xia, F., Huang, J., Zhu, L., Yang, Y., Lin, T. ChatLeafDisease: a chain-of-thought prompting approach for crop disease classification using large language models. Plant Phenomics, 7(3):100094, 2025. doi:10.1016/j.plaphe.2025.100094.',
  'Jin, D., Yin, H., Gu, Y.H. Shuffle-PG: Lightweight feature extraction model for retrieving images of plant diseases and pests with deep metric learning. Alexandria Engineering Journal, 113:138-149, 2025. doi:10.1016/j.aej.2024.11.052 (özet düzeyinde doğrulandı).',
  'Yang, Z., Wu, J., Yuan, X., Chen, Y., Guo, Y. General retrieval network model for multi-class plant leaf diseases based on hashing. PeerJ Computer Science, 10:e2545, 2024. doi:10.7717/peerj-cs.2545.',
  'Peng, Y., Wang, Y. Leaf disease image retrieval with object detection and deep metric learning. Frontiers in Plant Science, 13:963302, 2022. doi:10.3389/fpls.2022.963302.',
  'SegPPD-101 veri seti kaydı. Zenodo, 2025. doi:10.5281/zenodo.15114160.',
  'PlantSeg v2 veri seti kaydı. Zenodo, kayıt 13958858 (plantsegv2.zip). https://zenodo.org/records/13958858',
  'He, K., Zhang, X., Ren, S., Sun, J. Deep residual learning for image recognition. CVPR, 2016. doi:10.1109/CVPR.2016.90',
  'Oquab, M. ve ark. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. arXiv:2304.07193',
  'Khosla, P. ve ark. Supervised contrastive learning. NeurIPS, 2020. arXiv:2004.11362',
  'Deng, J., Guo, J., Xue, N., Zafeiriou, S. ArcFace: Additive angular margin loss for deep face recognition. CVPR, 2019. doi:10.1109/CVPR.2019.00482',
  'Kim, S. ve ark. Proxy anchor loss for deep metric learning. CVPR, 2020. arXiv:2003.13911'];
refs.forEach((r, i) => kids.push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}] ${r}`, font: FONT, size: 17 })], spacing: { after: 50 }, indent: { left: 360, hanging: 360 } })));

// ================= Ek
kids.push(H1('Ek. Ek Sıralama Deneyleri'));
kids.push(P('Bu bölüm, ana sonuçları değiştirmeyen ek sıralama deneylerini içerir. Bütün deneyler mevcut gömmelerle yapıldı, yeni eğitim yoktur. Seçim yalnızca geliştirme (dev) kümesinde, Hit@1 ile yapıldı (bir sorgu kadar tolerans bandı, bant içinde daha yüksek mAP); Protokol B ve C testleri daha önce kullanıldığı için bu kümelerdeki sayılar **keşifseldir**, yeni bağımsız test iddiası taşımaz.'));
kids.push(H2('Ek 1. Farklı uzaklık metrikleri'));
kids.push(P('Ana sistemde tam (exact) kare Öklid uzaklığı kullanılır. Karşılaştırma için kosinüs, iç çarpım, L1 (Manhattan) ve PCA beyazlatma + Öklid denendi. Beyazlatma yalnızca eğitim görüntülerinin gömmelerinden hesaplandı; kuvvet katsayısı α ∈ {0,25; 0,5; 1} ve boyut sayısı {256, 128, 64} ızgarası kullanıldı. Gömmeler L2-normalize olduğu için kosinüs ve iç çarpım sıralaması Öklid ile birebir aynıdır; bunu her hücrede doğruladık (aynı Hit@1). Altı model denendi: tek Proxy-Anchor modeli (3 tohum ortalaması), Proxy-Anchor üç tohum topluluğu, Bölüm 6.3\'teki SupCon üç tohum topluluğu (Öklid sonucu 0,822, Tablo 4 ile aynı) ve Tablo 4\'ün son satırı olan SupCon topluluğu + ResNet50 füzyonu (0,9 · topluluk + 0,1 · ince ayarlı ResNet50; Öklid sonucu 0,840, Tablo 4 ile aynı). Ayrıca, daha hafif bir seçenek olarak topluluk yerine TEK bir DINOv2-B modeline ResNet50 eklenen füzyon da denendi (Proxy-Anchor için w = 0,05, SupCon için w = 0,1; ağırlık yalnızca dev kümesinde seçildi; her tohum ayrı füzyonlanıp üç tohum ortalandı). Füzyonda parçalar L2-normalize edilip √ağırlık ile ölçeklenerek art arda eklenir; beyazlatma birleşik vektör üzerinde, eğitim gömmelerinden hesaplanır.'));
kids.push(CAPT('Tablo 8. Uzaklık metrikleri, Tablo 4 ile aynı ölçütlerle (Protokol B testi; 225 sorgu, galeri 809; keşifsel). Doğruluk = Hit@1; F1 makro; ROC-AUC sorgu-galeri çiftleri üzerinden, skor = −uzaklık. Tek model satırları 3 tohum ortalamasıdır; beyazlatma, o model için dev kümesinde seçilen ayardır.'));
kids.push(TABLE([3700, 1000, 850, 900, 800, 800, 950], ['Sistem', 'Doğruluk (Hit@1)', 'P@10', 'Hit@10', 'mAP', 'F1', 'ROC-AUC'], [
  ['Proxy-Anchor, tek model (3 tohum ort.), Öklid = kosinüs', '0,825', '0,656', '0,972', '0,753', '0,819', '0,991'],
  ['Proxy-Anchor, tek model, L1', '0,827', '0,655', '0,969', '0,751', '0,820', '0,990'],
  ['Proxy-Anchor, tek model, beyazlatma (α=0,25; 128 boyut)', '0,827', '0,660', '0,978', '0,757', '0,819', '0,991'],
  ['Proxy-Anchor, 3 tohum topluluğu, Öklid = kosinüs', '0,836', '0,668', '0,973', '0,766', '0,828', '0,993'],
  ['Proxy-Anchor topluluğu, L1', '0,831', '0,668', '0,973', '0,765', '0,822', '0,992'],
  ['Proxy-Anchor topluluğu, beyazlatma (α=0,25; 128 boyut)', '0,836', '0,670', '0,973', '0,768', '0,828', '0,992'],
  ['SupCon, 3 tohum topluluğu, Öklid = kosinüs (Tablo 4 ile aynı)', '0,822', '0,662', '0,969', '0,754', '0,812', '0,991'],
  ['SupCon topluluğu, L1', '0,822', '0,662', '0,969', '0,754', '0,812', '0,991'],
  ['SupCon topluluğu, beyazlatma (α=0,5; 64 boyut)', '0,836', '0,675', '0,969', '0,767', '0,820', '0,992'],
  ['SupCon topluluğu + ResNet50 füzyonu (w = 0,1), Öklid = kosinüs (Tablo 4 ile aynı)', '0,840', '0,664', '0,964', '0,757', '0,828', '0,991'],
  ['Topluluk + ResNet50 füzyonu, L1', '0,840', '0,666', '0,969', '0,757', '0,829', '0,991'],
  ['Topluluk + ResNet50 füzyonu, beyazlatma (α=0,25; 64 boyut)', '0,858', '0,664', '0,969', '0,763', '0,846', '0,992'],
  ['Proxy-Anchor TEK model + ResNet50 füzyonu (w = 0,05; 3 tohum ort.), Öklid = kosinüs', '0,839', '0,657', '0,972', '0,756', '0,830', '0,991'],
  ['Tek Proxy-Anchor + ResNet50, L1', '0,827', '0,655', '0,970', '0,753', '0,819', '0,991'],
  ['SupCon TEK model + ResNet50 füzyonu (w = 0,1; 3 tohum ort.), Öklid = kosinüs', '0,834', '0,653', '0,966', '0,746', '0,821', '0,990'],
  ['Tek SupCon + ResNet50, L1', '0,836', '0,651', '0,963', '0,743', '0,822', '0,990'],
  ['Tek SupCon + ResNet50, beyazlatma (α=0,25; 64 boyut)', '0,843', '0,657', '0,967', '0,752', '0,828', '0,991']], { size: 16, leftCols: 1 }));
kids.push(SP());
kids.push(CAPT('Tablo 9. Uzaklık metriklerinin karşılaştırması: Hit@1 (mAP). Sınıf-makro; B ve C testleri keşifsel. Beyazlatma satırı, o model için dev kümesinde seçilen ayardır.'));
kids.push(TABLE([2700, 2500, 1300, 1300, 1300], ['Model', 'Uzaklık', 'Dev (150)', 'B test (225)', 'C test (142)'], [
  ['Proxy-Anchor, tek model (3 tohum ort.)', 'Öklid = kosinüs', '0,820 (0,757)', '0,825 (0,753)', '0,782 (0,751)'],
  ['', 'L1', '0,818 (0,755)', '0,827 (0,751)', '0,791 (0,751)'],
  ['', 'Beyazlatma α=0,25, 128 boyut', '0,816 (0,760)', '0,827 (0,757)', '0,796 (0,756)'],
  ['Proxy-Anchor, 3 tohum topluluğu', 'Öklid = kosinüs', '0,813 (0,768)', '0,836 (0,766)', '0,813 (0,766)'],
  ['', 'L1', '0,800 (0,767)', '0,831 (0,765)', '0,813 (0,767)'],
  ['', 'Beyazlatma α=0,25, 128 boyut', '0,820 (0,771)', '0,836 (0,768)', '0,833 (0,772)'],
  ['SupCon, 3 tohum topluluğu', 'Öklid = kosinüs', '0,807 (0,761)', '0,822 (0,754)', '0,747 (0,739)'],
  ['', 'L1', '0,800 (0,761)', '0,822 (0,754)', '0,747 (0,739)'],
  ['', 'Beyazlatma α=0,5, 64 boyut', '0,827 (0,770)', '0,836 (0,767)', '0,747 (0,746)'],
  ['SupCon topluluğu + ResNet50 (w = 0,1)', 'Öklid = kosinüs', '0,820 (0,764)', '0,840 (0,757)', '0,773 (0,740)'],
  ['', 'L1', '0,820 (0,765)', '0,840 (0,757)', '0,760 (0,739)'],
  ['', 'Beyazlatma α=0,25, 64 boyut', '0,827 (0,771)', '0,858 (0,763)', '0,760 (0,744)'],
  ['Tek Proxy-Anchor + ResNet50 (w = 0,05)', 'Öklid = kosinüs (dev seçimi)', '0,827 (0,759)', '0,839 (0,756)', '0,800 (0,754)'],
  ['', 'L1', '0,818 (0,756)', '0,827 (0,753)', '0,782 (0,747)'],
  ['', 'Beyazlatma α=0,25, 128 boyut (seçilmedi)', '0,818 (0,760)', '0,831 (0,757)', '0,800 (0,756)'],
  ['Tek SupCon + ResNet50 (w = 0,1)', 'Öklid = kosinüs', '0,822 (0,755)', '0,834 (0,746)', '0,776 (0,735)'],
  ['', 'L1', '0,818 (0,756)', '0,836 (0,743)', '0,776 (0,732)'],
  ['', 'Beyazlatma α=0,25, 64 boyut (dev seçimi)', '0,827 (0,759)', '0,843 (0,752)', '0,769 (0,739)']], { size: 16, leftCols: 2 }));
kids.push(SP());
kids.push(P('Tablo 8 ve 9\'da görüldüğü gibi hiçbir metrik, geliştirme kümesinde Öklid\'in anlamlı biçimde üstüne çıkmamıştır. L1 hiçbir modelde kazanç sağlamamıştır (fark %95 güven aralıklarıyla sıfırı içerir). Beyazlatma, dev kümesinde birkaç ayarın bir sorgu toleransı içinde kalması ve mAP\'inin biraz yüksek olması nedeniyle seçilmiştir; Hit@1 kazancı nedeniyle değil. Seçilen beyazlatmanın Öklid\'e göre eşleştirilmiş farkı: tek Proxy-Anchor modelinde C testinde +0,013 [+0,004; +0,027], B testinde +0,002 [−0,009; +0,012]; Proxy-Anchor topluluğunda C testinde +0,020 [0,000; +0,047], B testinde 0,000; SupCon topluluğunda B testinde +0,013 [−0,013; +0,040], C testinde 0,000. Kazanç yalnızca bazı küme ve modellerde görülmekte, dev kümesinde görülmemekte ve 13 ayar denendiği için çoklu karşılaştırma düzeltmesi uygulanmamıştır; bu yüzden güvenilir bir iyileşme saymıyoruz. Topluluk + ResNet50 füzyonunda dev kümesi beyazlatma α=0,25, 64 boyutu seçti (dev +0,007 [0,000; +0,020]); bu ayar B testinde Hit@1\'i 0,858\'e çıkarmış (+0,018 [−0,009; +0,044]), C testinde ise Öklid\'in altında kalmıştır (0,760 karşı 0,773; fark −0,013 [−0,040; +0,013]); L1 füzyonda da kazanç getirmemiştir. Yani füzyonda da kazanç tutarsızdır ve güvenilir sayılmaz. Tam beyazlatma (α=1, 256 boyut) gürültülü yönleri büyüttüğü için belirgin biçimde kötüdür (tek modelde Hit@1 0,63-0,65). Sonuç olarak uzaklık metriği darboğaz değildir ve tam Öklid ana sistemde kalmıştır. Hocamızın "tam Öklid" şartı nedeniyle L1 ve iç çarpım yalnızca karşılaştırma amaçlıdır.'));
kids.push(P('Tek model + ResNet50 füzyonu (topluluksuz). Topluluk yerine tek bir DINOv2-B modeline ResNet50 eklendiğinde (Proxy-Anchor, w = 0,05) Öklid ile Hit@1 dev 0,827, B testi 0,839, C testi 0,800 elde edildi; tek modele göre eşleştirilmiş fark dev +0,007 [−0,004; +0,018], B +0,013 [+0,003; +0,025], C +0,018 [−0,004; +0,040] (mAP farkı her kümede ≤ +0,003). Bu füzyonda dev kuralı hiçbir alternatif metriği seçmedi, Öklid kaldı: L1 dev −0,009 [−0,018; −0,002], beyazlatma (α=0,25; 128 boyut) dev −0,009 [−0,020; +0,002]; testlerde de kazanç yok. SupCon tek model + ResNet50\'de (w = 0,1) dev kuralı beyazlatmayı (α=0,25; 64 boyut) seçti, ama Öklid\'e göre fark dev +0,004 [−0,009; +0,016], B +0,009 [−0,006; +0,025], C −0,007 [−0,016; +0,002], yani kazanç yoktur. Tablo 8\'in en yüksek hücresi olan topluluk + ResNet50 + beyazlatma (0,858), tek Proxy-Anchor + ResNet50 Öklid\'e göre B testinde +0,019 [−0,015; +0,053], C testinde −0,040 [−0,102; +0,018] verir: topluluğun tek füzyona üstünlüğü yoktur ve 0,858 değeri yalnızca B testine özgü, C testinde tekrarlanmayan bir sonuçtur. Tek füzyon, topluluk füzyonunun yaklaşık %38\'i kadar hesapla (27 GMAC karşı 71 GMAC; 110 M karşı 282 M parametre; gömme süresi 21 ms karşı 42 ms) aynı sonuç aralığındadır. B ve C tüketilmiş olduğundan bu sonuç da keşifseldir; doğrulanması için docs/FRESH_TEST_PLAN_PA_RESNET.md\'de tanımlanan, henüz koşulmamış taze test gerekir.'));
kids.push(H2('Ek 2. Yerel yama yeniden sıralaması (Chamfer)'));
kids.push(P('Küresel Öklid sıralamasının ilk M sonucu, DINOv2 son katman yama özniteliklerinin simetrik Chamfer benzerliği ile yeniden sıralandı (nihai puan = α · küresel kosinüs + (1 − α) · yerel benzerlik). Dev kuralı (Hit@1, ardından mAP) **yeniden sıralama yapmamayı** seçti: M=20, α=0,5 için dev farkı −0,004 [−0,042; +0,031]. Keşifsel kümelerde aynı ayar Hit@1\'i artırdı: B testinde +0,025 [+0,003; +0,049], C testinde +0,022 [−0,002; +0,049], harici PlantSeg\'de +0,018 [−0,012; +0,050]; mAP değişmedi. Sorgu başına ek gecikme M=20 için yaklaşık 1,3-1,4 ms, M=50 için 3,3-3,5 ms\'dir. Bu bir hipotezdir ve yeni, dokunulmamış bir test kümesinde doğrulanmalıdır.'));

// ================= Ek 3: parametre azaltma (research_v4)
function FIGPNG(file, widthPx, caption) {
  const buf = fs.readFileSync(file); const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20); const w = widthPx, h = Math.round(widthPx * ph / pw);
  return [new Paragraph({ children: [new ImageRun({ type: 'png', data: buf, transformation: { width: w, height: h }, altText: { title: file, description: caption, name: file } })], alignment: AlignmentType.CENTER, spacing: { before: 100, after: 0 }, keepNext: true }), CAPT(caption)];
}
kids.push(H2('Ek 3. Daha az parametreyle aynı doğruluk: parametre azaltma çalışması'));
kids.push(P('Tablo 4\'ün son satırındaki sistem (SupCon üç tohum topluluğu + ResNet50) çıkarımda 281,8 M parametre ve görüntü başına 71 GMAC gerektirir. Bu ek, aynı doğruluğu daha az parametreyle korumayı amaçlayan, önceden kayıt altına alınmış (artifacts/research_v4/registered.json) bir çalışmayı özetler. Hedef baştan "geliştirme, B ve C kümelerinin üçünde birden Hit@1 ≥ 0,85" olarak konuldu; Ek 1\'deki 0,858 değeri yalnızca B testine özgü olup C\'de tekrarlanmadığı için hedef sayılmadı. Seçim yalnızca geliştirme kümesinde yapıldı; B ve C sonuçları keşifseldir. Beş aşama denendi: (Z) eğitim gerektirmeyen son işlemler, (S) sekiz hafif omurganın aynı Proxy-Anchor tarifiyle taranması ve takip ayarları, (T) test zamanı çoklu görünüm, (D) üç tohum topluluğundan tek modele ilişkisel bilgi damıtma, (F) seçilen hafif adayın üç tohumla ölçümü.'));
kids.push(P('Sıfır parametreli yöntemler (Z): galeri tarafı artırma (DBA), sorgu genişletme (αQE), k-karşılıklı yeniden sıralama, yalnızca eğitim gömmelerinden öğrenilen NCA ve LDA izdüşümleri ile bunların dev kümesinde seçilen yığını. Hiçbiri tek Proxy-Anchor modelinde (dev 0,820) veya tek model + ResNet50 füzyonunda (dev 0,827) geliştirme Hit@1\'ini artırmadı; NCA belirgin kötüydü (0,78; 772 eğitim görüntüsü öğrenilmiş metrik için yetersiz), LDA B testinde +0,011 verirken dev ve C\'de kaybetti. Test zamanı çoklu görünüm (T; ölçek 1,25× ve beş kırpma) dev Hit@1\'i değiştirmedi, C testinde keşifsel +0,03 verdi, ama gömme süresini 7,5 kat artırdı. Damıtma (D; öğretmen = üç tohum topluluğu veya topluluk + ResNet50, kayıp = Proxy-Anchor + λ · parti benzerlik matrisleri arasındaki KOH) DINOv2-B öğrencide altı, ConvNeXt-T öğrencide dört koşunun hiçbirinde kazanç vermedi: tek model, topluluğun benzerlik yapısını zaten taşımaktadır.'));
kids.push(CAPT('Tablo 10. Omurga taraması (aynı Proxy-Anchor tarifi; seçim dev kümesinde; B ve C keşifsel). Hit@1 (mAP). Tohum 0, final satırları 3 tohum ortalaması.'));
kids.push(TABLE([3300, 1000, 1300, 1200, 1200, 900], ['Omurga (tarif)', 'Parametre', 'Dev', 'B test', 'C test', 'VRAM'], [
  ['DINOv2-B son 4 blok (referans, 3 tohum ort.)', '85,9 M', '0,820 (0,757)', '0,825 (0,753)', '0,782 (0,751)', '2,98 GB'],
  ['DINOv3 ViT-B/16 son 4 blok', '85,8 M', '0,793 (0,665)', '0,800 (0,664)', '0,760 (0,677)', '2,75 GB'],
  ['DINOv2-B 9 bloğa kesilmiş', '64,7 M', '0,713 (0,587)', '0,671', '0,647', '2,89 GB'],
  ['DINOv2-B 8 bloğa kesilmiş', '57,6 M', '0,687 (0,548)', '0,631', '0,613', '2,87 GB'],
  ['DINOv3 ConvNeXt-T, 224 px', '28,0 M', '0,773 (0,684)', '0,689', '0,713', '0,98 GB'],
  ['DINOv3 ConvNeXt-T, 336 px (tohum 0)', '28,0 M', '0,793 (0,686)', '0,711', '0,693', '~2,5 GB'],
  ['DINOv3 ConvNeXt-T, 336 px, 3 tohum ort. (final)', '28,0 M', '0,773 ± 0,041 (0,674)', '0,702 (0,634)', '0,691 (0,646)', '1,80 GB'],
  ['DINOv3 ViT-S+/16', '28,8 M', '0,733 (0,570)', '0,747', '0,693', '1,64 GB'],
  ['DINOv3 ViT-S/16', '21,7 M', '0,747 (0,523)', '0,698', '0,640', '1,22 GB'],
  ['DINOv2-S reg4, 224 px', '21,7 M', '0,760 (0,636)', '0,756', '0,660', '1,35 GB'],
  ['DINOv2-S reg4, 336 px', '21,9 M', '0,780 (0,674)', '0,747', '0,733', '2,77 GB'],
  ['DINOv2-S (kayıtsız)', '21,7 M', '0,713 (0,535)', '0,649', '0,620', '1,34 GB']], { size: 16, leftCols: 1 }));
kids.push(SP());
kids.push(...FIGPNG('report/figs/P1_params_vs_hit1.png', 560, 'Şekil 12. Çıkarım parametresi ile Hit@1: geliştirme kümesi (sol) ve B+C birleşik, keşifsel (sağ). Daireler bu çalışmadaki tüm eğitilmiş adaylar (tohum 0; tam ince ayar, damıtma ve takip ayarları dahil), elmaslar önceki sistemler ve üç tohumlu final. Kesikli çizgi hedef 0,85.'));
kids.push(P('Sonuç. Hedef 0,85 hiçbir sistemle, hafif ya da ağır, üç kümede birden sağlanamadı. Kullanılabilir sonuç, doğruluğu koruyan parametre azaltmasıdır: topluluk + ResNet50 (281,8 M) yerine tek DINOv2-B + Proxy-Anchor (85,9 M; 3,3 kat az parametre, 3,2 kat az hesap) aynı sonuç aralığındadır; istenirse tek model + ResNet50 füzyonu (109,9 M) B testinde +0,013 [+0,003; +0,025] verir (keşifsel). Yaklaşık 86 M\'nin altına inildiğinde doğruluk güvenilir biçimde düşer: en iyi 28 M model (ConvNeXt-T, 336 px) referansa göre dev −0,047 [−0,082; −0,009], B −0,123 [−0,163; −0,083], C −0,091 [−0,149; −0,031] kaybeder ve tohumlar arası sapması dört kat büyüktür; derinlik budama daha da kötüdür; iki hafif modelin füzyonu farkı kapatmaz. Doğruluğu artıracak olan mimari değil veridir: 772 eğitim görüntüsü bütün omurgaları dev kümesinde 0,79-0,82 bandında sınırlamaktadır; çapraz kaynak eğitimi (SegPPD + PlantSeg) ve dokunulmamış bir taze test kümesi sonraki adımlardır. Ayrıntılı tablolar docs/PARAMETER_REDUCTION_STUDY.md ve artifacts/research_v4/ altındadır.'));

const doc = new Document({
  creator: 'SegPPD-101 CBIR', title: 'SegPPD-101 CBIR Araştırma Raporu',
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 260, after: 110 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 22, bold: true }, paragraph: { spacing: { before: 180, after: 90 }, outlineLevel: 1 } }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1304, right: 1304 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
    children: kids }]
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync('docs/CBIR_RESEARCH_REPORT.docx', b); console.log('written', b.length); });
