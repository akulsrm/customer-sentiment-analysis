import pandas as pd
import numpy as np

class SentimentDataGenerator:
    def __init__(self):
        self.positive_templates = [
            "This {product} is absolutely amazing! {positive_adj}",
            "I love this {product}! It's {positive_adj} and {positive_adj}.",
            "Excellent {product}! Highly recommend to everyone.",
            "Best {product} I've ever {action}. {positive_adj} quality!",
            "Outstanding {product}! Exceeded my expectations.",
            "Fantastic {product}! Worth every penny.",
            "Great {product}! {positive_adj} and {positive_adj}.",
            "Perfect {product}! Couldn't be happier with my purchase.",
            "Amazing {product}! {positive_adj} design and functionality.",
            "Wonderful {product}! Excellent customer service too."
        ]
        
        self.negative_templates = [
            "This {product} is terrible! {negative_adj} and {negative_adj}.",
            "Very disappointed with this {product}. Poor quality.",
            "Worst {product} I've ever {action}. Complete waste of money.",
            "Horrible {product}! Would not recommend to anyone.",
            "Awful {product}! {negative_adj} construction and materials.",
            "Poor {product}! Broke after just {time_period}.",
            "Cheap {product}! {negative_adj} quality and design.",
            "Useless {product}! Doesn't work as advertised.",
            "Regret buying this {product}. Very {negative_adj}.",
            "Terrible {product}! Customer service was also {negative_adj}."
        ]
        
        self.neutral_templates = [
            "This {product} is okay. Nothing special but works fine.",
            "Average {product}. Does what it's supposed to do.",
            "The {product} is fine for the price. Standard quality.",
            "Decent {product}. Meets basic expectations.",
            "Standard {product}. Nothing to complain about.",
            "Fair {product}. Acceptable quality and functionality.",
            "The {product} works as expected. Nothing extraordinary.",
            "Okay {product}. Could be better, could be worse.",
            "Basic {product}. Gets the job done.",
            "Regular {product}. Average experience overall."
        ]
        
        self.positive_words = ['excellent', 'amazing', 'fantastic', 'wonderful', 'great', 
                              'perfect', 'outstanding', 'superb', 'incredible', 'awesome']
        
        self.negative_words = ['terrible', 'awful', 'horrible', 'poor', 'cheap', 
                              'useless', 'disappointing', 'bad', 'worse', 'worst']
        
        self.products = ['product', 'item', 'purchase', 'buy', 'acquisition']
        
        self.actions = ['bought', 'purchased', 'acquired', 'got', 'ordered']
        
        self.time_periods = ['one day', 'a week', 'a month', 'a few days', 'a short time']
    
    def generate_sentiment_data(self, num_samples_per_class=200):
        """Generate comprehensive sentiment dataset"""
        data = []
        
        # Generate positive reviews
        for i in range(num_samples_per_class):
            template = np.random.choice(self.positive_templates)
            product = np.random.choice(self.products)
            action = np.random.choice(self.actions)
            positive_adj = np.random.choice(self.positive_words)
            second_adj = np.random.choice(self.positive_words)
            
            review = template.format(
                product=product,
                action=action,
                positive_adj=positive_adj
            ).replace('{positive_adj}', second_adj)
            
            data.append({'review': review, 'sentiment': 1})
        
        # Generate negative reviews
        for i in range(num_samples_per_class):
            template = np.random.choice(self.negative_templates)
            product = np.random.choice(self.products)
            action = np.random.choice(self.actions)
            negative_adj = np.random.choice(self.negative_words)
            second_adj = np.random.choice(self.negative_words)
            time_period = np.random.choice(self.time_periods)
            
            review = template.format(
                product=product,
                action=action,
                negative_adj=negative_adj,
                time_period=time_period
            ).replace('{negative_adj}', second_adj)
            
            data.append({'review': review, 'sentiment': 0})
        
        # Generate neutral reviews
        for i in range(num_samples_per_class):
            template = np.random.choice(self.neutral_templates)
            product = np.random.choice(self.products)
            
            review = template.format(product=product)
            
            data.append({'review': review, 'sentiment': 2})
        
        # Add some real-world examples
        real_examples = [
            ("I absolutely love this product! It exceeded all my expectations and the quality is outstanding.", 1),
            ("This is the worst purchase I've ever made. Complete waste of money and terrible quality.", 0),
            ("The product arrived on time and works as described. Nothing special but does the job.", 2),
            ("Amazing quality and fast shipping! Customer service was also excellent.", 1),
            ("Very disappointed. The product broke after just one day of use.", 0),
            ("It's an okay product. Fair price for what you get.", 2),
            ("Fantastic! Best product in its category. Highly recommend!", 1),
            ("Poor construction and cheap materials. Would not buy again.", 0),
            ("Average product with standard features. Decent value for money.", 2),
            ("Perfect! Exactly what I was looking for. Great quality and design.", 1)
        ]
        
        for review, sentiment in real_examples:
            data.append({'review': review, 'sentiment': sentiment})
        
        # Shuffle the data
        np.random.shuffle(data)
        
        return pd.DataFrame(data)
    
    def save_dataset(self, filename='sentiment_dataset.csv', num_samples_per_class=200):
        """Generate and save dataset to CSV"""
        print(f"Generating sentiment dataset with {num_samples_per_class} samples per class...")
        df = self.generate_sentiment_data(num_samples_per_class)
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(df)}")
        print(f"Class distribution:")
        print(df['sentiment'].value_counts().sort_index())
        return df

if __name__ == "__main__":
    generator = SentimentDataGenerator()
    df = generator.save_dataset(num_samples_per_class=300)  # Generate 900+ samples total