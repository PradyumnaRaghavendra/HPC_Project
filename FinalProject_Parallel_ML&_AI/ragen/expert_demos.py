"""
Expert demonstrations for WebShop
Hand-crafted perfect trajectories showing the model how to succeed
"""

def get_expert_demos():
    """
    Perfect trajectories across diverse product categories.
    These teach the model the action sequence: search → click → buy
    """
    return [
        # Electronics & Accessories
        {
            'instruction': 'find blue wireless headphones',
            'actions': [
                'search[blue wireless headphones]',
                'click[B09QKP7XQL]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find iphone charging cable',
            'actions': [
                'search[iphone charging cable]',
                'click[B08FXYZ123]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find digital camera black',
            'actions': [
                'search[digital camera black]',
                'click[B09CAM456]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        
        # Clothing - Women
        {
            'instruction': 'find women sweaters',
            'actions': [
                'search[women sweaters]',
                'click[B08Y865MTQ]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find women dresses polyester',
            'actions': [
                'search[women dresses polyester]',
                'click[B07DRESS99]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find women hoodie white',
            'actions': [
                'search[women hoodie white]',
                'click[B09HOOD123]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        
        # Clothing - Men
        {
            'instruction': 'find men tuxedo shirt',
            'actions': [
                'search[men tuxedo shirt]',
                'click[B08SHIRT77]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find men henley shirt',
            'actions': [
                'search[men henley shirt]',
                'click[B09HENLEY1]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find men sneakers',
            'actions': [
                'search[men sneakers]',
                'click[B07SHOE999]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        
        # Beauty & Personal Care
        {
            'instruction': 'find makeup brushes',
            'actions': [
                'search[makeup brushes]',
                'click[B08BRUSH22]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find shampoo damaged hair',
            'actions': [
                'search[shampoo damaged hair]',
                'click[B01LOUY5M8]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find cruelty free deodorant',
            'actions': [
                'search[cruelty free deodorant]',
                'click[B09DEOD456]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        
        # Food & Snacks
        {
            'instruction': 'find gluten free snacks',
            'actions': [
                'search[gluten free snacks]',
                'click[B08SNACK11]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find organic chocolate',
            'actions': [
                'search[organic chocolate]',
                'click[B07CHOC789]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        
        # Home & Kitchen
        {
            'instruction': 'find refillable water bottle',
            'actions': [
                'search[refillable water bottle]',
                'click[B09BOTTLE5]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find living room furniture',
            'actions': [
                'search[living room furniture]',
                'click[B08FURN999]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find stainless steel cookware',
            'actions': [
                'search[stainless steel cookware]',
                'click[B09COOK456]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find coffee maker black',
            'actions': [
                'search[coffee maker black]',
                'click[B07COFFEE1]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Sports & Outdoors
        {
            'instruction': 'find yoga mat non-slip',
            'actions': [
                'search[yoga mat non-slip]',
                'click[B08YOGA789]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find running shoes men',
            'actions': [
                'search[running shoes men]',
                'click[B09RUNSHOE]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find camping tent waterproof',
            'actions': [
                'search[camping tent waterproof]',
                'click[B07TENT999]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find basketball',
            'actions': [
                'search[basketball]',
                'click[B08BALL123]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Books & Media
        {
            'instruction': 'find science fiction book',
            'actions': [
                'search[science fiction book]',
                'click[B09SCIFI88]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find mystery novel paperback',
            'actions': [
                'search[mystery novel paperback]',
                'click[B07MYSTER1]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Toys & Games
        {
            'instruction': 'find building blocks kids',
            'actions': [
                'search[building blocks kids]',
                'click[B08BLOCKS9]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find board game family',
            'actions': [
                'search[board game family]',
                'click[B09GAME456]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find puzzle 1000 pieces',
            'actions': [
                'search[puzzle 1000 pieces]',
                'click[B08PUZZLE7]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Pet Supplies
        {
            'instruction': 'find dog food grain free',
            'actions': [
                'search[dog food grain free]',
                'click[B09DOGFOOD]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find cat litter box',
            'actions': [
                'search[cat litter box]',
                'click[B07CATLITT]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find pet carrier small',
            'actions': [
                'search[pet carrier small]',
                'click[B08CARRIER]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Office Supplies
        {
            'instruction': 'find notebook lined paper',
            'actions': [
                'search[notebook lined paper]',
                'click[B09NOTE123]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find printer ink black',
            'actions': [
                'search[printer ink black]',
                'click[B08INK7890]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find desk organizer',
            'actions': [
                'search[desk organizer]',
                'click[B07DESKORG]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find mechanical pencil',
            'actions': [
                'search[mechanical pencil]',
                'click[B09PENCIL5]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Automotive
        {
            'instruction': 'find car phone mount',
            'actions': [
                'search[car phone mount]',
                'click[B08MOUNT99]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find windshield wiper blades',
            'actions': [
                'search[windshield wiper blades]',
                'click[B07WIPER12]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Health & Wellness
        {
            'instruction': 'find vitamins multivitamin',
            'actions': [
                'search[vitamins multivitamin]',
                'click[B09VIT7890]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find resistance bands workout',
            'actions': [
                'search[resistance bands workout]',
                'click[B08BANDS56]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find protein powder chocolate',
            'actions': [
                'search[protein powder chocolate]',
                'click[B09PROTEIN]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Garden & Outdoor
        {
            'instruction': 'find garden hose expandable',
            'actions': [
                'search[garden hose expandable]',
                'click[B08HOSE789]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find plant pots ceramic',
            'actions': [
                'search[plant pots ceramic]',
                'click[B07POTS123]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Baby & Kids
        {
            'instruction': 'find baby bottles bpa free',
            'actions': [
                'search[baby bottles bpa free]',
                'click[B09BOTTLE8]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find diaper bag large',
            'actions': [
                'search[diaper bag large]',
                'click[B08DIAPER1]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find baby monitor video',
            'actions': [
                'search[baby monitor video]',
                'click[B09MONITOR]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Musical Instruments
        {
            'instruction': 'find guitar strings acoustic',
            'actions': [
                'search[guitar strings acoustic]',
                'click[B07STRING9]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find keyboard piano beginner',
            'actions': [
                'search[keyboard piano beginner]',
                'click[B09PIANO88]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Jewelry & Accessories
        {
            'instruction': 'find silver necklace women',
            'actions': [
                'search[silver necklace women]',
                'click[B08NECK789]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find watch leather band',
            'actions': [
                'search[watch leather band]',
                'click[B09WATCH12]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find sunglasses polarized',
            'actions': [
                'search[sunglasses polarized]',
                'click[B07SUNGLAS]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Luggage & Travel
        {
            'instruction': 'find backpack travel laptop',
            'actions': [
                'search[backpack travel laptop]',
                'click[B09BACKPAK]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find suitcase hardside spinner',
            'actions': [
                'search[suitcase hardside spinner]',
                'click[B08SUITCAS]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Art & Crafts
        {
            'instruction': 'find acrylic paint set',
            'actions': [
                'search[acrylic paint set]',
                'click[B09PAINT56]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find sketchbook drawing',
            'actions': [
                'search[sketchbook drawing]',
                'click[B07SKETCH8]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },

        # Additional varied examples for robustness
        {
            'instruction': 'find red backpack',
            'actions': [
                'search[red backpack]',
                'click[B09REDBAG1]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find cotton t-shirt',
            'actions': [
                'search[cotton t-shirt]',
                'click[B08TSHIRT7]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
        {
            'instruction': 'find white sneakers',
            'actions': [
                'search[white sneakers]',
                'click[B09WHITESH]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],
            'total_reward': 1.0,
            'success': True
        },
    ]


def format_demos_for_sft():
    """
    Format expert demonstrations as (prompt, completion) pairs for SFT.
    This is used for behavior cloning phase before RL training.
    """
    demos = get_expert_demos()
    sft_data = []

    for demo in demos:
        instruction = demo['instruction']
        actions = demo['actions']

        # Create training examples for each action
        for i, action in enumerate(actions):
            # Simulate the observation the model would see
            if i == 0:
                # Initial state - just the task
                obs = f"Task: {instruction}\n\nInstruction:\n\nSearch"
            elif 'search' in actions[i-1]:
                # After search - show product list (simulated)
                obs = f"Task: {instruction}\n\nBack to Search\n\nPage 1 (Total results: 10) [SEP] Next > [SEP] B08X2FSR21 [SEP] Product Name [SEP] $29.99 [SEP] B09ABC123 [SEP] Another Product [SEP] $35.99"
            elif 'click' in actions[i-1]:
                # After click - show product details
                obs = f"Product: Perfect Match\nPrice: $29.99\nThis matches your search perfectly!"
            else:
                obs = f"Task: {instruction}"

            prompt = f"<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n"
            completion = action

            sft_data.append((prompt, completion))

    return sft_data


def get_partial_demos():
    """
    Demonstrations showing PARTIAL success.
    These teach intermediate steps.
    """
    return [
        # Just searching (teaches search format)
        {
            'instruction': 'find shoes',
            'actions': [
                'search[shoes]',
                'back'
            ],
            'rewards': [0.0, 0.0],
            'total_reward': 0.0,
            'success': False,
            'partial_credit': 0.1  # Reward for valid search
        },
        # Searching + clicking (teaches click)
        {
            'instruction': 'find laptop',
            'actions': [
                'search[laptop]',
                'click[B09LAPTOP1]',
                'back'
            ],
            'rewards': [0.0, 0.0, 0.0],
            'total_reward': 0.0,
            'success': False,
            'partial_credit': 0.25  # Reward for search + click
        },
    ]