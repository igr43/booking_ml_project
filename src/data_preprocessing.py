import pandas as pd

def preprocess_and_feature_engineering(df):
    """
    Предобработка данных и создание новых признаков
    """
    print(" ПРЕДОБРАБОТКА ДАННЫХ И FEATURE ENGINEERING")
    print("=" * 50)
    
    df_processed = df.copy()
    
    # 1. Преобразование дат
    if 'review_date' in df_processed.columns:
        df_processed['review_date'] = pd.to_datetime(df_processed['review_date'])
        df_processed['review_year'] = df_processed['review_date'].dt.year
        df_processed['review_month'] = df_processed['review_date'].dt.month
        df_processed['review_day'] = df_processed['review_date'].dt.day
        df_processed['review_dayofweek'] = df_processed['review_date'].dt.dayofweek
        
        # Сезонность
        df_processed['season'] = df_processed['review_month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
                     'Spring' if x in [3, 4, 5] else
                     'Summer' if x in [6, 7, 8] else 'Autumn'
        )
        print("Добавлены временные признаки")
    
    # 2. Разрывы между базовыми и реальными оценками
    if all(col in df_processed.columns for col in ['score_cleanliness', 'cleanliness_base']):
        df_processed['cleanliness_gap'] = df_processed['score_cleanliness'] - df_processed['cleanliness_base']
        print("Добавлен cleanliness_gap")
    
    if all(col in df_processed.columns for col in ['score_comfort', 'comfort_base']):
        df_processed['comfort_gap'] = df_processed['score_comfort'] - df_processed['comfort_base']
        print("Добавлен comfort_gap")
    
    if all(col in df_processed.columns for col in ['score_facilities', 'facilities_base']):
        df_processed['facilities_gap'] = df_processed['score_facilities'] - df_processed['facilities_base']
        print("Добавлен facilities_gap")
    
    # 3. Агрегаты на уровне отеля (средние оценки отеля)
    if 'hotel_id' in df_processed.columns and 'score_overall' in df_processed.columns:
        hotel_agg = df_processed.groupby('hotel_id').agg({
            'score_overall': ['mean', 'count'],
            'score_cleanliness': 'mean',
            'score_comfort': 'mean',
            'score_staff': 'mean'
        }).round(3)
        
        hotel_agg.columns = ['hotel_avg_score', 'hotel_review_count', 
                           'hotel_avg_cleanliness', 'hotel_avg_comfort', 'hotel_avg_staff']
        
        df_processed = pd.merge(df_processed, hotel_agg, on='hotel_id', how='left')
        print(" Добавлены агрегированные признаки отеля")
    
    # 4. Дополнительные признаки
    df_processed['total_specific_scores'] = df_processed[['score_cleanliness', 'score_comfort', 
                                                         'score_facilities', 'score_location', 
                                                         'score_staff']].sum(axis=1)
    
    df_processed['score_variance'] = df_processed[['score_cleanliness', 'score_comfort', 
                                                  'score_facilities', 'score_location', 
                                                  'score_staff']].var(axis=1)
    
    print(f"    Итоговый размер после обработки: {df_processed.shape}")
    print(f"    Новые колонки: {[col for col in df_processed.columns if col not in df.columns]}")
    
    return df_processed

