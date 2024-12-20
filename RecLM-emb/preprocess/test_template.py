# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

user2item_template = [
    "I have the following histories: {}.",
    "Recommend an item for me based on my history interactions: {}.",
    "I have played several games before: {}. Can you recommend other games for me?",
    "{} are games I have just played, are there any other games suitable for me?",
    "Given the following item history {}, predict next possible item to be played?",
    "Here are the history games a user has played {}. What to recommend next for the user?",
    "Find a game for me based on these previous games: {}.",
    "I've played these games before: {}. What other games should I try?",
    "Played games list: {}. Suggest a new game for me to play.",
    "My gaming history includes: {}. What game should I play next?",
    "These are the games I've experienced: {}. Offer some new recommendations.",
    "My game history: {}. Find a suitable game for me.",
    "Played these games: {}. What's a good game to try next?",
    "Considering my previous games {}, what would you recommend?",
    "My game background includes: {}. Any recommendations?",
    "Games I've played before: {}. What should I play now?",
    "Played and enjoyed these games: {}. What's next?",
    "Here's my gaming history: {}. Can you suggest a new game?",
    "Check out my game history {}. What game do you recommend?",
    "Histories: {}",
    "I have played {}.",
    "Enjoyed games: {}.",
    "Played games: {}. Recommend a new game for me.",
    "I've played these games: {}.",
    "{} are games I've played. Any recommendations?",
    "History interactions include: {}.",
    "These are the past games I've engaged in: {}. Can you suggest another game?",
    "Based on my game-play history {}, what would be a good game to play next?",
    "Here's a list of games I've previously played: {}. Any suggestions for what to play next?",
    "I've had experience playing the following games: {}. What do you recommend I play next?",
    "My past gaming experiences include: {}. What game should I try out next?",
    "My gaming past involves: {}. Suggest another game I might enjoy.",
    "These are the games I've participated in: {}. What should be my next game?",
    "I've spent time playing these games: {}. Can you find a suitable next game for me?",
    "I have history with these games: {}. What's a good game to move on to?",
    "Having played these games: {}, what would be a good next step?",
    "Based on these games I've enjoyed: {}, what should I play next?",
    "Here are some games I've previously engaged in: {}. What do you propose I play next?",
    "I've had fun playing these games: {}. What's the next game you would suggest?",
    "Having enjoyed these games: {}, which game should I tackle next?",
    "These are games from my past: {}. Can you recommend a new one for me?",
    "Given my gaming history {}, what would you suggest I play next?",
    "Here's a list of games I've enjoyed: {}. What should I try next?",
    "Look at the games I've previously played: {}. Could you recommend a new one?",
    "These are games from my history: {}. What's a good next game?",
    "I've spent some time with these games: {}. What's a suitable next game?",
]

unorder_user2item_template = [
    "I have the following no chronological histories: {}.",
    "Played these games (unordered): {}. What's a good game to try for me?",
    "Recommend an item for me based on my mixed history interactions: {}.",
    "I have played several games before (in no particular order): {}. Can you recommend other games for me?",
    "{} are games I have just played (shuffled). Are there any other interesting games you would recommend?",
    "Given the following non-sequential item history {}, what games would you recommend I play?",
    "Find some fun games for me based on these previous ones (jumbled): {}.",
    "I've played these games before (not in sequence): {}. What other games should I try?",
    "Disorganized played games list: {}. Suggest a new game for me to play.",
    "My gaming history includes these scattered games: {}. Could you propose some games for me?",
    "These are the games I've experienced (shuffled): {}. Offer some new recommendations.",
    "My game history (not in order): {}. Find a suitable game for me.",
    "Played these games (unsorted): {}. What games would you suggest for me?",
    "Considering my previous games (in no particular order) {}, what would you recommend?",
    "Non-sequential histories: {}. Can you find some suitable games for me?",
    "I have played these shuffled games: {}.",
    "Enjoyed games (in no specific order): {}.",
    "Played games (unordered): {}. Recommend a new game for me.",
    "I've played these games (scattered): {}.",
    "{} are games I've played no chronological order. Any recommendations?",
    "Shuffled history interactions include: {}.",
    "Unordered history interactions include: {}.",
    "I have the following random histories: {}. Can you recommend some games for me?",
    "Recommend an item for me based on my mixed history interactions: {}.",
    "Enjoyed games (in no specific order): {}. What other games might be fun?",
    "These are the past games I've engaged in (unsorted): {}. Any suggestions for new ones?",
    "Based on my game-play history (shuffled): {}, what games might be worth checking out?",
    "My past gaming experiences include these unordered games: {}. What other games could you suggest?",
    "My gaming past involves these unsorted games: {}. What games do you think I'd enjoy?",
    "These are the games I've participated in (in no chronological order): {}. Could you suggest a few more games?",
    "Having enjoyed these games (non-sequential): {}, could you recommend some additional games?",
    "Look at the games I've previously played (jumbled): {}. Could you find some more games?",
    "These are games from my history (unsorted): {}. What other games might suit my taste?", 
    "I've had fun playing these unsorted games: {}. What other games might be enjoyable?",
]

query2item_template = [
    "{}",
    "I want to find some games with several features: {}.", 
    "Recommend games for a user who likes games that have these features {}.", 
    "I prefer games with the following type: {}, can you recommend some for me?",
    "Recommend an item that meets the following requirements: {}.",
    "I'm looking for games that include these elements: {}.",
    "Suggest some games for someone who enjoys features like {}.",
    "I have a taste for this kind of games {}, what do you recommend?",
    "Find me a game that matches these criteria: {}.",
    "I'm in search of games with these characteristics: {}.",
    "What games can you suggest that offer these features? {}",
    "I'd like to try games that incorporate {}. Any recommendations?",
    "Help me find games with the following attributes: {}.",
    "Can you recommend games that focus on elements like {}",
    "I enjoy games with {}, can you suggest any?",
    "I'm interested in games that provide elements like {}. What are my options?",
    "What are some good games that emphasize {}?",
    "I'd love to discover games that have {}. Any suggestions?",
    "I'm a fan of this style games: {}. Can you offer some recommendations?",
    "Looking for game suggestions with qualities such as {}",
    "I'm curious about games that highlight {}. What can you recommend?",
    "Games containing {}.",
    "I want to find games featuring {}.",
    "Recommend games with {}.",
    "I'm looking for games that have {}.",
    "I'm interested in games that include {}.",
    "{} are features I'm looking for.",
    "query: {}",
    "Search for games with {}.",
    "Search for {}",
    "I'm on the hunt for games that possess these attributes: {}.",
    "Can you suggest any games that are built around these elements: {}?",
    "I'm fascinated by games that feature {}. Can you provide some suggestions?",
    "Help me locate games that contain these specific aspects: {}.",
    "I'm a fan of games with these characteristics: {}. Any recommendations?",
    "Could you recommend games that spotlight {}?",
    "I'm seeking games that involve {}. Any proposals?",
    "I'm into games that concentrate on {}. Can you list some for me?",
    "I'm exploring games that emphasize {}. Any thoughts?",
    "Please suggest games that are rich in these features: {}.",
    "I'd appreciate if you could recommend games that encapsulate {}.",
    "I'm keen on games that offer {}. Can you guide me to some?",
    "I'm after games that focus on {}. Any ideas?",
    "Can you propose games that cater to these particularities: {}",
    "I'd like to explore games that involve {}. Any tips?",
    "What games could you recommend that integrate these elements {}?",
    "I'm fond of games with {}. Any advice?",
    "I'm in pursuit of games with {}. Any leads?",
    "I'm intrigued by games that entail {}. What are your recommendations?",
    "I'm eager to find games that showcase {}. Any suggestions?",
    "Looking for games that possess these features {}.",
    "I'm interested in games that are characterized by {}.",
    "Help me locate games with these features {}.",
    "Seeking game recommendations with these elements {}.",
    "I'd like to discover games that utilize {}.",
    "Search for games that highlight {}.",
    "I'm in quest of games that exemplify {}. Any recommendations?",
    "Can you list games that include elements such as {}?",
    "I'm after games that embody {}. What can you suggest?",
    "I'm passionate about games that exhibit {}. Any ideas?",
]

title2item_template = [
    "{}",
    "I'm looking for this game: {}.",
    "I want to find this game: {}.",
    "I'm looking for a game called {}.",
    "Search for this game: {}.",
    "I'm trying to find this game: {}.",
    "Please help me locate this game: {}.",
    "Can you find this game for me: {}?",
    "I'm in search of a game named {}.",
    "Could you search for a game titled {}?",
    "I need assistance finding this game: {}.",
    "I'm interested in locating a game called {}.",
    "Please look up this game for me: {}.",
    "I'd like to locate the game {}.",
    "Help me find this game, please: {}.",
    "I'm on the hunt for a game named {}.",
    "Could you assist me in locating this game: {}?",
    "I'm eager to find this specific game: {}.",
    "I'm trying to track down a game called {}.",
    "Can you assist in finding this game: {}?",
    "I'm in pursuit of a game titled {}.",
    "I'm seeking a game known as {}.",
    "Please assist me in tracking down this game: {}.",
    "I'm attempting to locate this game: {}.",
    "Would you help me in finding this game: {}?",
    "I'm trying to pinpoint the game {}.",
    "I have been searching for a game called {}.",
    "Can you help me discover this game: {}?",
    "I'm in need of help to find this game: {}.",
    "Please aid me in my search for this game: {}.",
    "I'm keen on finding a game titled {}.",
    "Could you lend a hand in locating this game: {}?",
    "I'm scouting for a game named {}.",
    "Help me locate this game, if you can: {}.",
    "I wish to find this particular game: {}.",
]

item2item_template = [
    "Can you recommend me some items similar to {}?",
    "I have played this game: {} and find it interesting. Are there other games similar to this one?",
    "Looking for games like this: {}. Any suggestions?",
    "I enjoyed this item: {}, can you recommend anything similar?",
    "Any recommendations for games related to {}?",
    "Find me more games like {} please.",
    "Which games share similarities with this game? {}",
    "Just played this game: {}, and I'm craving more with the same feel. Any ideas?",
    "{} was so much fun! What other similar games would you recommend?",
    "Are there any other games with the same feel as this game? {}",
    "Games similar to this one {}",
    "I'm looking for games similar to {}",
    "Recommend games similar to this game: {}.",
    "Similar games to {}.",
    "Games like {}.",
    "Items similar to {}.",
    "Items like {}.",
    "Recommend items similar to this one: {}.",
    "Related games to {}.",
    "Could you suggest some products that resemble {}?",
    "I've enjoyed playing this game: {}. Any other games with a similar vibe you could suggest?",
    "In search of games akin to this: {}. Any ideas?",
    "I had a great time with this item: {}, any recommendations for something similar?",
    "Do you have any suggestions for games linked to {}?",
    "Could you fetch me more games parallel to {}?",
    "What games bear resemblance to this one? {}",
    "Recently played this game: {}, longing for more of the same kind. Any suggestions?",
    "{} was an absolute blast! Could you suggest other games along the same lines?",
    "Are there any games that provide a similar experience as this game? {}",
    "Games that match up to this one {}",
    "I'm in search of games that mirror {}",
    "Could you suggest games that are equivalent to this game: {}.",
    "Games akin to {}.",
    "Games reminiscent of {}.",
    "Products parallel to {}.",
    "Goods similar to {}.",
    "Could you suggest goods that are alike with this one: {}.",
    "Games connected to {}.",
]

queryuser2item_template = [
    "target: {1},  histories: {0}",
    "histories: {0}, target: {1}",
    "I want to find some games with several features: {1}. These are games I have played before: {0}.",
    "Given the following item history {0}, predict the next possible item meeting the following requirements: {1}",
    "Based on my gaming history {0}, suggest some games that have these attributes: {1}.",
    "Considering my previous games {0}, please recommend new games possessing these qualities: {1}.",
    "Using my game history {0} as reference, provide game suggestions with the following characteristics: {1}.",
    "I'm looking for games with these features: {1}. Here are games I've played before: {0}.",
    "Show me games with these traits: {1}, taking into account my past games {0}.",
    "In light of the games I've enjoyed {0}, suggest new games that include these aspects: {1}.",
    "Discover games that have the following attributes: {1}, based on my previous gaming experience {0}.",
    "Find games that contain these properties: {1}, keeping my game history in mind: {0}.",
    "Search for games with these elements: {1}, considering my gaming background {0}.",
    "With my past games {0} as a guide, present new games featuring these characteristics: {1}.",
    "Given my previous gaming choices {0}, I'd like to explore games that offer these features: {1}.",
    "My past gaming experience {0} leads me to seek games with these characteristics: {1}.",
    "Following my game history {0}, kindly suggest games with these specific features: {1}.",
    "Please provide game recommendations with these qualities: {1}, in relation to my gaming past {0}.",
    "features: {1}, history interactions: {0}",
    "history interactions: {0}, features: {1}",
    "histories: {0}, features: {1}",
    "features: {1}, histories: {0}",
    "I have played {0}, recommend games with features: {1}.",
    "Characteristics: {1}, games I have played: {0}.",
    "Look for games contain {1}, based on my gaming history {0}.",
    "Enjoy features: {1}. Played games: {0}.",
    "Played games: {0}. Enjoy features: {1}.",
    "Formerly played: {0}, seeking games with attributes: {1}",
    "Attributes sought: {1}, based on past games: {0}",
    "I have a history of playing these games: {0}, suggest games that include these elements: {1}.",
    "Given my game history {0}, I'd appreciate game suggestions that encompass these features: {1}.",
    "In view of my past games {0}, I'm in search of games that contain these characteristics: {1}.",
    "Using my gaming history {0} as a basis, suggest games with these particular traits: {1}.",
    "I'm seeking games with these properties: {1}, considering my past gaming experience: {0}.",
    "Show me games with these distinctive aspects: {1}, keeping my gaming history {0} in mind.",
    "In consideration of the games I've played {0}, recommend new games with these features: {1}.",
    "Find games with these criteria: {1}, based on my gaming past {0}.",
    "Search for games with these qualities: {1}, referencing my prior game history {0}.",
    "Considering my gaming background {0}, recommend games that align with these characteristics: {1}.",
    "With my game history {0} as a reference, suggest games with these specific elements: {1}.",
    "Given my gaming interactions {0}, I'd like to find games that exhibit these traits: {1}.",
    "My gaming history {0} prompts me to look for games with these aspects: {1}.",
    "Following my game interactions {0}, please suggest games that possess these features: {1}.",
    "Please offer game suggestions with these characteristics: {1}, in accordance to my game history {0}.",
    "Desired traits: {1}, past game interactions: {0}",
    "Past game interactions: {0}, desired traits: {1}",
    "Interactions history: {0}, desired properties: {1}",
    "Desired properties: {1}, interactions history: {0}",
    "I have experienced {0}, recommend games with these qualities: {1}.",
    "Attributes: {1}, past games: {0}.",
    "Identify games with {1}, considering my game interactions {0}.",
    "Interested in features: {1}. Games experienced: {0}.",
    "Games experienced: {0}. Interested in features: {1}.",
]

querysummary2item_template = [
    "user summary: {0}, query: {1} ",
    "Here is summary of the user history: {0}, following this query: {1}",
    "user query: {1}, summary of user preference: {0}",
    "search request: {1}, User's background: {0}",
    "Overview of user's interests: {0}, related inquiry: {1}",
    "User profile highlights: {0}, associated question: {1}",
    "User's main interests: {0}, search query: {1}",
    "Summary of user's past activities: {0}, with this search request: {1}",
    "Query from user: {1}, user's preference summary: {0}",
    "Research question: {1}, User's personal interest: {0}",
    "Synopsis of user's favorites: {0}, pertinent search: {1}",
    "User's preference snapshot: {0}, linked query: {1}",
    "User's activity summary: {0}, subsequent inquiry: {1}",
    "Here is a brief of user's likes: {0}, following the question: {1}",
    "User's inquiry: {1}, recap of user's likes: {0}",
    "Search question: {1}, User's preference overview: {0}",
    "User's interests summary: {0}, related search request: {1}",
    "User's profile summary: {0}, connected query: {1}",
    "User's query: {1}, summary of user's interests: {0}",
    "User's search: {1}, Background of user's preference: {0}",
    "Interests of the user: {0}, corresponding search: {1}",
    "Highlights of user's profile: {0}, linked search: {1}",
    "Summary of the user's likes: {0}, following this inquiry: {1}",
    "User's question: {1}, summary of user's past activities: {0}",
    "User's search request: {1}, User's preference synopsis: {0}",
    "Interest's summary of the user: {0}, associated search: {1}"
]

vaguequery2item_template = [
    "{}",
    "I want to find some games with several features: {}.", 
    "Recommend games for a user who likes games that have these features: {}.", 
    "I prefer games with the following type: {}, can you recommend some for me?",
    "Recommend an item that meets the following requirements: {}.",
    "I'm looking for games that include these elements: {}.",
    "Suggest some games for someone who enjoys features like {}.",
    "Find me a game that matches these criteria: {}.",
    "I'm in search of games with these characteristics: {}.",
    "What games can you suggest that offer these features? {}",
    "Help me find games with the following attributes: {}.",
    "I'm interested in games that provide these elements: {}. What are my options?",
    "Games: {}.",
    "I want to find games with {}.",
    "Recommend games with {}.",
    "I'm looking for games that have {}.",
    "I'm interested in games that include {}.",
    "query: {}",
    "Search for games: {}.",
    "I am on the hunt for games incorporating these aspects: {}.",
    "Can you suggest some games that encompass these features: {}?",
    "I'm in pursuit of games that embody these qualities: {}.",
    "Games that align with these specifications: {}, could you list some?",
    "Recommend a game that fulfills these criteria: {}.",
    "I'm scouting for games that encapsulate these components: {}.",
    "Propose some games for a player who appreciates attributes such as {}.",
    "Locate a game for me that adheres to these stipulations: {}.",
    "I'm on a quest for games with these particularities: {}.",
    "What games could you put forward that encompass these characteristics? {}",
    "Aid me in locating games with these particular traits: {}.",
    "I'm captivated by games that comprise these elements: {}. Can you list some options?",
    "Games with these features: {}.",
    "I'm keen on finding games with {}.",
    "Suggest games that possess {}.",
    "I am scouting for games that feature {}.",
    "I have a fascination for games that incorporate {}.",
    "Search query: {}",
    "Look up for games that contain: {}.",
    "Inquiry for games with these specifications: {}."
]

relativequery2item_template = {
    "recent":[
        "Recommend me some recently released games.",
        "Could you suggest some new games that have just been launched?",
        "Can you tell me about some of the latest games on the market?",
        "I'm looking for some newly released games, can you help?",
        "What are some of the newest games that have just come out?",
        "Can you list some of the latest video games that have been released recently?",
        "Do you have any recommendations for games that have been released lately?",
        "I'm interested in the latest games, could you provide some suggestions?",
        "Could you provide some information on the latest games that have just hit the market?",
        "I'm seeking some new games that have recently been launched, can you assist?",
        "Can you name a few of the most recent games that are now available?",
        "Are there any newly debuted games that you can suggest?",
        "What are some of the most recent video games that have just been released?",
        "Can you inform me about the latest games that are currently on sale?",
        "Do you have any suggestions for the latest released games that are worth playing?",
        "I'm curious about the newest games, could you offer some recommendations?",
        "Could you tell me about the latest games that have just been released?",
        "I'm in search of some new games that have just come out, can you help?",
        "Can you introduce me to some of the most recent games on the market?",
        "Are there any recommendations for games that have been launched recently?",
        "I'm keen on the latest games, can you provide some suggestions?",
        "Can you give me a list of the latest video games that have just been released?",
        "Do you have any ideas about games that have just hit the market?",
        "I'm on the lookout for the latest games, can you give some advice?",
        "Can you share some information on the newest games that have just been launched?",
        "What are some of the latest games that are currently available?",
        "Can you help me find some of the most recently released video games?",
        "Do you have any thoughts on games that have been released lately?"
    ],
    "cheap":[
        "Recommend me some cheap games.",
        "I'm looking for some cheap games, can you help?",
        "What are some of the cheapest games you can recommend?",
        "Could you suggest some inexpensive games?",
        "Can you tell me about some of the most affordable games?",
        "I'm interested in the cheapest games, could you provide some suggestions?",
        "Could you suggest some affordable games for me?", 
        "Can you help me find some budget-friendly games?", 
        "I'm looking for some low-cost games, can you assist?", 
        "What are some cost-effective games that I could buy?", 
        "Can you list some of the inexpensive games available?", 
        "Do you have any recommendations for games that are cheap?", 
        "I'm interested in games that won't break the bank, could you provide some suggestions?",
        "Could you provide some suggestions for games that are reasonably priced?",
        "I'm seeking some budget games, can you assist?",
        "Can you name a few of the most cost-effective games currently available?",
        "Are there any low-priced games that you can recommend?",
        "What are some of the most economical games that you can suggest?",
        "Can you inform me about the cheapest games that are currently on sale?",
        "Do you have any suggestions for the best value games?",
        "I'm curious about the most affordable games, could you offer some options?",
        "Could you tell me about the best games for a small budget?",
        "I'm in search of some games that are light on the wallet, can you help?",
        "Can you introduce me to some of the most economically priced games?",
        "Are there any recommendations for games that are a bargain?",
        "I'm keen on finding games that are a steal, can you provide some suggestions?",
        "Can you give me a list of the most inexpensive games currently available?",
        "Do you have any ideas about games that are priced low?",
        "I'm on the lookout for games that are easy on the pocket, can you give some advice?",
        "Can you share some information on the best games that won't cost a lot?",
        "What are some of the best games that are a good deal?",
        "Can you help me find some of the most budget-conscious games?",
        "Do you have any thoughts on games that are a great value for the price?"
    ],
    "expensive":[
        "Recommend me some expensive games.",
        "I'm looking for some expensive games, can you help?",
        "What are some of the most expensive games you can recommend?",
        "Could you suggest some pricey games?",
        "Can you tell me about some of the most costly games?",
        "I'm interested in the most expensive games, could you provide some suggestions?",
        "Can you help me find some high-priced games?", 
        "I'm looking for some high-cost games, can you assist?", 
        "Can you list some of the expensive games available?", 
        "Do you have any recommendations for games that are costly?", 
        "Can you help me find some premium-priced games?",
        "I'm looking for some costly games, can you assist?",
        "What are some of the high-priced games that I could buy?",
        "Can you list some of the pricy games available?",
        "Do you have any recommendations for games that are expensive?",
        "I'm interested in luxury games, could you provide some suggestions?",
        "Could you suggest some high-end games?",
        "I'm in search of some top-tier priced games, could you help?",
        "What are some luxury-priced games you could recommend?",
        "Can you advise on some high-value games?",
        "Could you provide details about some of the priciest games?",
        "I'm curious about top-priced games, could you provide some recommendations?",
        "Can you assist me in finding some steep-priced games?",
        "I'm seeking out some luxury games, can you aid?",
        "Could you enumerate some of the expensive games on the market?",
        "Do you have any suggestions for games with a high price tag?",
        "Can you help me identify some exorbitantly priced games?",
        "I'm on the hunt for some steeply priced games, can you support?",
        "What are some of the top-dollar games that I could purchase?",
        "Could you display some of the expensive games in circulation?",
        "Do you have any tips for games with a hefty price?",
        "I have a keen interest in top-of-the-line games, could you provide some recommendations?",
        "Could you propose some extravagant games?",
        "I'm exploring some of the priciest games, can you guide?",
        "Can you showcase some of the costliest games available?",
        "I'm fascinated by premium games, could you offer some suggestions?"
    ],
    "popular":[
        "Recommend me some popular games.",
        "I'm looking for some popular games, can you help?",
        "What are some of the most popular games you can recommend?",
        "Could you suggest some popular games?",
        "Can you tell me about some of the most popular games?",
        "I'm interested in the most popular games, could you provide some suggestions?",
        "Can you help me find some popular games?", 
        "I'm looking for some popular games, can you assist?", 
        "Can you list some of the popular games available?", 
        "Do you have any recommendations for games that are popular?", 
        "I'm interested in games that are trending, could you provide some suggestions?",
        "Could you suggest some trending games for me?",
        "Can you help me find some of the most played games right now?",
        "I'm looking for some hit games, can you assist?",
        "What are some of the hot games that are popular right now?",
        "Can you list some of the most downloaded games at the moment?",
        "Do you have any recommendations for games that are currently popular?",
        "I'm interested in mainstream games, could you provide some suggestions?",
        "Could you point me towards some trending games?",
        "I'm seeking out some popular games, could you lend a hand?",
        "What are some widely played games that you would suggest?",
        "Would you mind recommending some popular games?",
        "Can you inform me about some of the hot games in the market?",
        "I'm keen on the trending games, could you offer some recommendations?",
        "Can you aid me in discovering some popular games?",
        "I'm on the hunt for some popular games, can you provide support?",
        "Could you enumerate some of the crowd-pleasing games currently?",
        "Do you hold any suggestions for games that are in vogue?",
        "I'm fascinated by games that are in the limelight, could you offer some advice?",
        "Could you propose some games that are gaining popularity for me?",
        "Can you assist me in locating some of the most engaged games currently?",
        "I'm scouting for some blockbuster games, could you help?",
        "What are some of the buzzworthy games that are gaining popularity now?",
        "Can you catalog some of the most frequently downloaded games these days?",
        "Do you carry any endorsements for games that are presently in demand?",
        "I'm engrossed in mainstream games, could you put forward some options?",
        "I'm hunting for the top-rated games, can you guide me?",
        "Can you share a list of games that are currently making waves?"
    ],
}

negquery2item_template = [
    "not including {}",
    "Games excluding these features: {}",
    "I want to find some games excluding several features: {}.", 
    "Recommend games for a user who likes games that don't have these features: {}.", 
    "Recommend an item that meets the following requirements: not including {}.",
    "I'm looking for games that don't include these elements: {}.",
    "Find me a game that matches these criteria: excluding {}.",
    "I'd love to discover games that do not have {}. Any suggestions?",
    "Games not containing {}.",
    "I want to find games not featuring {}.",
    "Recommend games excluding these elements: {}.",
    "I'm looking for games that do not have {}.",
    "I'm interested in games that do not include {}.",
    "Search for games excluding these features: {}.",
    "Search for games not including {}",
    "Avoiding these features: {}",
    "Games without these characteristics: {}",
    "I'm in search of games that miss out on these features: {}.",
    "Suggest games for someone who prefers games lacking these elements: {}.",
    "Propose an item that abides by these conditions: devoid of {}.",
    "I'm on the hunt for games that aren't inclusive of these aspects: {}.",
    "Locate a game for me that fits these specifics: devoid of {}.",
    "I'd be thrilled to explore games that are absent of {}. Any proposals?",
    "Games devoid of {}.",
    "I aim to discover games devoid of {}.",
    "Propose games that eliminate these components: {}.",
    "I'm in pursuit of games that are free of {}.",
    "I'm captivated by games that lack {}.",
    "Look up games that omit these features: {}.",
    "Seek out games that don't incorporate {}",
    "Games that steer clear of these features: {}",
    "I desire games that are not packed with these elements: {}",
    "Suggest an object that aligns with these prerequisites: missing {}",
    "I'm scouting for games that don't carry these attributes: {}",
    "Could you find a game that satisfies these standards: missing {}"
]

