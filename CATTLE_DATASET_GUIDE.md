# 🐄 Cattle Disease Dataset Structure

## 📁 **Your Image Folders** (Ready to Use!)

```
data/raw/
├── healthy/                    ← Healthy cattle images
├── mastitis/                   ← Udder infection (most common)
├── foot_and_mouth_disease/     ← Viral disease affecting hooves/mouth  
├── lumpy_skin_disease/         ← Viral skin nodules
├── dermatophilosis/           ← Bacterial skin infection
└── ringworm/                  ← Fungal skin infection
```

## 🎯 **Focus on These First** (Best Dataset Availability):

### 1. **Mastitis** 🥛
- **Most Common**: Udder infection in dairy cattle
- **Visual Signs**: Swollen, red udders; abnormal milk
- **Dataset Availability**: ⭐⭐⭐⭐⭐ (Excellent)
- **AI Training**: Easy to classify visually

### 2. **Lumpy Skin Disease** 🦠
- **Distinctive**: Characteristic skin nodules
- **Visual Signs**: Raised lumps 2-5cm diameter
- **Dataset Availability**: ⭐⭐⭐⭐ (Very Good)
- **AI Training**: Excellent visual features

### 3. **Healthy Cattle** ✅
- **Baseline**: Normal, disease-free animals
- **Visual Signs**: Good body condition, clear eyes/nose
- **Dataset Availability**: ⭐⭐⭐⭐⭐ (Excellent)
- **AI Training**: Essential for comparison

## 📊 **Recommended Minimum Images per Category**:

| Disease Category | Minimum Images | Ideal Images |
|------------------|---------------|--------------|
| Healthy | 100+ | 200-500 |
| Mastitis | 100+ | 200-500 |
| Lumpy Skin Disease | 50+ | 100-300 |
| Foot & Mouth Disease | 50+ | 100-300 |
| Others | 30+ | 50-200 |

## 🎯 **Quick Start Recommendation**:

**Start with just 3 categories for best results:**
1. `healthy/` - 100+ images
2. `mastitis/` - 100+ images  
3. `lumpy_skin_disease/` - 50+ images

This gives you:
- ✅ Balanced dataset
- ✅ Clear visual differences
- ✅ Common diseases with good data availability
- ✅ Easier to find training images

## 🔍 **Where to Find Cattle Disease Images**:

1. **Research Papers** - Look for veterinary journals
2. **Agricultural Universities** - Often have public datasets
3. **Veterinary Colleges** - May share educational materials
4. **Government Agriculture Departments** - Disease surveillance data
5. **Kaggle/Academic Datasets** - Search for "cattle disease", "livestock health"

## 🚀 **After Adding Your Images**:

```bash
# 1. Check your data
python launch.py info

# 2. Preprocess the dataset  
python launch.py preprocess

# 3. Train the AI model
python launch.py train

# 4. Test the system
python launch.py web
```

## 💡 **Pro Tips**:

- **Image Quality**: Clear, well-lit photos work best
- **Variety**: Different angles, breeds, ages, lighting
- **Focus Areas**: Close-ups of affected areas (udders, skin, hooves)
- **Balanced Data**: Similar number of images per category
- **Validation**: Make sure images are correctly labeled

---

**You're ready to build your cattle disease AI! 🚀**

Just add your images to the folders above and run the training pipeline!