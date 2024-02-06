# Karelia-Project

Idea of this Karelian project was to perform zero-shot information retrival on stories about Karelian people. Stories are based on books "Siirtokarjalaisten tie" and there is University of Helsinki database called "Learning-from-our-past". Altho the database includes a lot of data about these people we where intrested in the source texts. Our task was to differentiate social organizations as well as hobbies for primary person (the one being interviewed) and spouse respectively. 

Stories are each different and they hold consist pattern in the beginning, but lack that in the ladder aspect where hobbies and social organizations are discussed. Husband or wife can both be interviewed. This lack of consistant pattern raises the challenge why we decided to utilize LLMs for this task. Because the text actually needs to be read in order to understand which hobbie or organization belong to whom, since it is not always clear by names, or other keywords, when the narrative changes. There is total of 89,339 of source texts to process. For this size of dataset we decided it is worth it to pay GPT-4 which performs substantially better than any other model to this day. 

It is worth to mention that while GPT-4s performance ecceeds our expectations with a whopping F score "0.96". We still managed to train a NER-tagger based on Fin-BERT utilizing this GPT-4 data. This NER method gave us F score of "0.91" already with "~6000" data samples. Altho this seemed to be the ceiling for this method. Larger training sets didn't give us better results. 

***Disclaimer: Numbers presented here are not yet verified *** 


API 

- Includes our scripts for GPT-4 data processing. As well as script for running Llama-2-70B-Chat 

Sample of our annotated data is in format like this. 

{
        "index": 2,
        "primary_person_name": "TOIVO RAVANTTI",
        "primary_person_id": "siirtokarjalaiset_2_17390P",
        "spouse_name": "Aino Inkeri Luukka",
        "spouse_id": "siirtokarjalaiset_2_17390S_1",
        "source_text": "toim.johtaja, synt. 22. 6. -29 Antreassa. Puol. Aino Inkeri o.s. Luukka, liikeapul., synt. 28. 9. -29 Uudellakirkolla. Lapset: Jukka Tapani -58 Riihimäki, Jarmo Juhani 59 Riihimäki, Jari Olavi -62 Loppi. Asuinp. Karjalassa: Antrea, Kaltove-denmaankylä 39, 42—44. Muut asuinp : Tammela -42, Ikaalinen 44—, Janakkala, Loppi, Sajaniemi 47—54, Loppi, Pilpala 54—58, Riihimäki, Loppi kk. 60—. Toivo Ravantti asuu perheineen omakotitalossaan. Toim.johtaja Ravantti on jäsenenä Lopen Karjalaiset ry.ssä. verolautakunnassa ja Sajaniemen Hirviveikot ry:ssä. Hänen harrastuksenaan on metsästys ja kalastus. Rouva kuuluu jäsenenä Lopen Karjalaiset ry:een ja toimii Lopen Kuparsaaren Marttojen sihteerinä. Hänen harrastuksenaan ovat käsityöt ja puutarhanhoito.",
        "person_hobbies": "metsästys, kalastus",
        "person_social_orgs": "Lopen Karjalaiset ry, verolautakunta, Sajaniemien hirviveikot ry",
        "spouse_hobbies": "käsityöt, puutarhanhoito",
        "spouse_social_orgs": "Lopen Karjalaiset ry, Lopen Kuparsaaren Martat"
}

Name Entity Recognition

- Here are the methods that takes GPT-4 results and searches a match from the original source text utilising Levenshtein distance. When the match is found these tokens are then labeled to corresponding categories. P-HOB, P-ORG, S-HOB, S-ORG or 'o' for outside. 

- After this process we use this data to train NER-tagger. 

Data Scripts

- Here we have miccelaneous scripts utilised to process the data from one to another. 
