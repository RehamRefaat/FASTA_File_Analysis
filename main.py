import time
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
from Bio import pairwise2, AlignIO, SeqIO
import pandas as pd # to read file
import matplotlib.pyplot as plt # for creating animated and interactive visualizations
from IPython.display import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#get_ipython().run_line_magic('matplotlib', 'inline')

RNA_Codons_seqtoprotien = {
            # 'M' - START, '*' - STOP
            "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "UGU": "C", "UGC": "C",
            "GAU": "D", "GAC": "D",
            "GAA": "E", "GAG": "E",
            "UUU": "F", "UUC": "F",
            "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
            "CAU": "H", "CAC": "H",
            "AUA": "I", "AUU": "I", "AUC": "I",
            "AAA": "K", "AAG": "K",
            "UUA": "L", "UUG": "L", "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
            "AUG": "M",
            "AAU": "N", "AAC": "N",
            "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "CAA": "Q", "CAG": "Q",
            "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
            "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", "AGU": "S", "AGC": "S",
            "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
            "UGG": "W",
            "UAU": "Y", "UAC": "Y",
            "UAA": "*", "UAG": "*", "UGA": "*"
        }
RNA_Codons_protientoseq = {
    # 'M' - START, '*' - STOP
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "C": ["UGU", "UGC"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["UUU", "UUC"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUA", "AUU", "AUC"],
    "K": ["AAA", "AAG"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "M": ["AUG"],
    "N": ["AAU", "AAC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "*": ["UAA", "UAG", "UGA"]
}
proteinweight = {"A": 89, "V": 117, "L": 131, "I": 131, "P": 115, "F": 165, "W": 204, "M": 149, "G": 75,
                             "S": 105, "C": 121, "T": 119, "Y": 181, "N": 132, "Q": 146, "D": 133, "E": 147, "K": 146,
                             "R": 174, "H": 155}
def Dictionary(Pseq):
    d = dict([
        ('A', Pseq.count('A')),
        ('C', Pseq.count('C')),
        ('D', Pseq.count('D')),
        ('E', Pseq.count('E')),
        ('F', Pseq.count('F')),
        ('G', Pseq.count('G')),
        ('H', Pseq.count('H')),
        ('I', Pseq.count('I')),
        ('K', Pseq.count('K')),
        ('L', Pseq.count('L')),
        ('M', Pseq.count('M')),
        ('N', Pseq.count('N')),
        ('P', Pseq.count('P')),
        ('Q', Pseq.count('Q')),
        ('R', Pseq.count('R')),
        ('S', Pseq.count('S')),
        ('T', Pseq.count('T')),
        ('V', Pseq.count('V')),
        ('W', Pseq.count('W')),
        ('Y', Pseq.count('Y')),
        ('*', Pseq.count('*'))])
    return d
def ProtienWeight(Pseq):
    totalweight = 0  # هنحسب التوتال ويت
    for x in Pseq:
        totalweight = totalweight + proteinweight.get(x, 0)
    totalweight = totalweight - (18 * (len(Pseq) - 1))
    return totalweight
def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Protein Analysis", "DNA Analysis","Classifying DNA With ML"],#دي الاوبسشن اللي هتكون عندي في المنيو
        icons=["house", "file-code", "activity","bootstrap"])#بتحط لكل اوبشن الايكون بتاعة
if selected == "Home":
    st.header("Welcome to our FASTA Analysis webpage 👋")
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json)
if selected == "Protein Analysis":
    lottie_url = "https://assets1.lottiefiles.com/packages/lf20_u6fnid8x.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json)
    selected = option_menu(
        menu_title=None,
        options=["UniProtKB", "UniRef", "UniParc"],
        orientation="horizontal")#المنيو اللي بختار منها نوع فاستا بتاعت البروتين
    if selected == "UniProtKB":
        UniProtKBEntry = st.text_area("Enter your FASTA of Protein from UniProtKB", "Type Here")#متغير بيخزن النص الفاستا اللي تم ادخاله في صفحة الويب في صوره استرنج
        if st.checkbox("Submit UniProtKB"):
            st.balloons()
            with st.spinner('Wait for it...'):
                time.sleep(2)
            #copy = UniProtKBEntry  # عملت منه نسخه علشان لو عاوزة استخدمه بعد لما اعدل عليه

            dictionary1 = {}
            #ابدايه من هنا هبدا اقسم الفاستا واحطها في ديكشناري
            #[P53_HUMAN, Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
            #MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
            UniProtKBEntry = UniProtKBEntry[1:]  #بشيل علامة <
            UniProtKBEntry = UniProtKBEntry.split("|", 2)
            dictionary1["db"] = [UniProtKBEntry[0]]
            dictionary1["Unique Identifier"] = [UniProtKBEntry[1]]
            UniProtKBEntry = UniProtKBEntry[2].split()
            dictionary1["Entry Name"] = [UniProtKBEntry[0]]
            UniProtKBEntry = " ".join(UniProtKBEntry)
            start = UniProtKBEntry.find(" ")
            end = UniProtKBEntry.find("OS=")
            proteinName = UniProtKBEntry[start + 1:end - 1:1]
            dictionary1["Protein Name"] = [proteinName]
            start = UniProtKBEntry.find("OS=")
            end = UniProtKBEntry.find("OX=")
            OS = UniProtKBEntry[start + 3:end - 1:1]
            dictionary1["Organism Name"] = [OS]
            start = UniProtKBEntry.find("OX=")
            end = UniProtKBEntry.find("GN=")
            OX = UniProtKBEntry[start + 3:end - 1:1]
            dictionary1["Organism Identifier"] = [OX]
            start = UniProtKBEntry.find("GN=")
            end = UniProtKBEntry.find("PE=")
            GN = UniProtKBEntry[start + 3:end - 1:1]
            dictionary1["Gene Name"] = [GN]
            start = UniProtKBEntry.find("PE=")
            end = UniProtKBEntry.find("SV=")
            PE = UniProtKBEntry[start + 3:end - 1:1]
            dictionary1["Protein Existence"] = [PE]
            start = UniProtKBEntry.find("SV=")
            end = UniProtKBEntry.find(" ", start)
            SV = UniProtKBEntry[start + 3:end:1]
            dictionary1["Sequence Version"] = [SV]
            # ********************************************************************************
            df = pd.DataFrame.from_dict(dictionary1, orient="index", columns=["DATA"])#بحول الديكشنري للي عندي الي داتا فريم
            st.dataframe(df)#تعرضلي المعلومات بتاعت الفاستا بعد لما قسمتهم علي شكل داتا فريم
            # ********************************************************************************
            st.subheader("Protein Sequences")
            s = UniProtKBEntry.find("SV=")
            start = UniProtKBEntry.find(" ",s)
            proteinseq = UniProtKBEntry[start+1::1]
            st.success(proteinseq)#بتعرض سيكونس البروتين الموجود في الفاستا
            #********************************************************************************
            st.subheader("The Protein Weight")
            totalweight = ProtienWeight(proteinseq)#فانكشن ببعتلها سيكونس البروتين بتحسبلي الويت وترجعه
            st.text(proteinweight)#البروتين ويت ده متغير نوعة ديكشناري مخزنه فيه الويت بتاعت البروتين
            st.success("The net weight = " + str(totalweight))
            # ********************************************************************************
            st.subheader("Number Of Protiens")
            d1=Dictionary(proteinseq)#بانده فانكشن وابعتلها البروتين بتعد عدد البروتينات وتجعهملي في ديكشناري
            dfcount = pd.DataFrame.from_dict(d1, orient="index", columns=["Number"])#داتافريم خاصة بعدد البروتينات
            st.dataframe(dfcount)
            # ********************************************************************************
            st.header("Select Visualization")
            options = st.selectbox("",["Visualization the Number of Protein","Visualization of Protein Weight"])
            if options == "Visualization the Number of Protein":
                st.subheader("Select type of  Visualization")
                op = st.selectbox("",["Bar","Line","Area","Pie"])
                if op == "Bar":
                    st.bar_chart(dfcount)
                if op == "Line":
                    st.line_chart(dfcount)
                if op == "Area":
                    st.area_chart(dfcount)
                if op == "Pie":
                    newdict={}#عملت واحد جديد علشان اشيل القيم اللي بصفر علشان يكون الشكل مظبوط
                    for x in d1:
                        if d1[x] > 0:
                            newdict[x] = d1[x]
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(newdict.values(), labels=newdict.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            if options == "Visualization of Protein Weight":
                st.subheader("Select type of Visualization")
                dfweight = pd.DataFrame.from_dict(proteinweight, orient="index",columns=["Number"])  # داتافريم خاصة وزن البروتين
                op=st.selectbox("",["Bar","Line","Area","Pie"])
                if op == "Bar":
                    st.bar_chart(dfweight)
                if op == "Line":
                    st.line_chart(dfweight)
                if op == "Area":
                    st.area_chart(dfweight)
                if op == "Pie":
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(proteinweight.values(), labels=proteinweight.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            # ********************************************************************************
            st.subheader("Possible Sequences")
            proteinseq = proteinseq.replace("\n",'')
            proteinseq = proteinseq.replace(' ','')
            protein ={}
            #MEEPQ
            for i in range(0, len(proteinseq)):#باخد البروتينات اللي عندي في فاستا واشوف ايه السيكونس المتوقع ليها
                protein[proteinseq[i]] = RNA_Codons_protientoseq[proteinseq[i]]
            st.write(protein)
    if selected == "UniRef":
        UniRefEntry = st.text_area("Enter your FASTA of Protein from UniRef ", "Type Here")# متغير بيخزن النص الفاستا اللي تم ادخاله في صفحة الويب في صوره استرنج
        if st.checkbox("Submit UniRef"):
            st.balloons()
            with st.spinner('Wait for it...'):
                time.sleep(2)
            dictionary2 = {}
            copy = UniRefEntry
            UniRefEntry=UniRefEntry[1:]
            end=UniRefEntry.find(" ")
            dictionary2["Unique Identifier"] = [UniRefEntry[:end]]
            start=UniRefEntry.find(" ")
            end=UniRefEntry.find("n=")
            dictionary2["Cluster Name"] = [UniRefEntry[start+1:end-1]]
            start=UniRefEntry.find("n=")
            end=UniRefEntry.find("Tax=")
            dictionary2["Members"] = [UniRefEntry[start+2:end-1]]
            start = UniRefEntry.find("Tax=")
            end = UniRefEntry.find("TaxID=")
            dictionary2["Taxon Name"] = [UniRefEntry[start +4:end - 1]]
            start = UniRefEntry.find("TaxID=")
            end = UniRefEntry.find("RepID=")
            dictionary2["Taxon Identifier"] = [UniRefEntry[start + 6:end - 1]]
            start = UniRefEntry.find("RepID=")
            end = UniRefEntry.find("\n")
            dictionary2["Representative Member"] = [UniRefEntry[start + 6:end]]
            # ********************************************************************************
            df = pd.DataFrame.from_dict(dictionary2, orient="index", columns=["DATA"])# بحول الديكشنري للي عندي الي داتا فريم
            st.dataframe(df)#تعرضلي المعلومات بتاعت الفاستا بعد لما قسمتهم
            # ********************************************************************************
            st.subheader("Protein Sequences")
            #s = UniRefEntry.find("RepID=")
            start = UniRefEntry.find("\n")
            proteinseq1 = UniRefEntry[start::1]
            st.success(proteinseq1)#بتعرض سيكونس البروتين الموجود في الفاستا
            # ********************************************************************************
            st.subheader("The Protein Weight")
            totalweight = ProtienWeight(proteinseq1) # فانكشن ببعتلها سيكونس البروتين بتحسبلي الويت وترجعه
            st.text(proteinweight) # البروتين ويت ده متغير نوعة ديكشناري مخزنه فيه الويت بتاعت البروتين
            st.success("The net weight = " + str(totalweight))
            # ********************************************************************************
            st.subheader("Number Of Protiens")
            d2 = Dictionary(proteinseq1) #بانده فانكشن وابعتلها البروتين بتعد عدد البروتينات وتجعهملي في ديكشناري
            dfcount = pd.DataFrame.from_dict(d2, orient="index", columns=["Number"])# داتافريم خاصة بعدد البروتينات
            st.dataframe(dfcount)
            # ********************************************************************************
            st.header("Select Visualization")
            options = st.selectbox("", ["Visualization the Number of Protein", "Visualization of Protein Weight"])
            if options == "Visualization the Number of Protein":
                st.subheader("Select type of  Visualization")
                op=st.selectbox("",["Bar","Line","Area","Pie"])
                if op == "Bar":
                    st.bar_chart(dfcount)
                if op == "Line":
                    st.line_chart(dfcount)
                if op == "Area":
                    st.area_chart(dfcount)
                if op == "Pie":
                    newdict={}#ملت واحد جديد علشان اشيل القيم اللي بصفر علشان يكون الشكل مظبوط
                    for x in d2:
                        if d2[x] > 0:
                            newdict[x]=d2[x]
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(newdict.values(), labels=newdict.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            if options=="Visualization of Protein Weight":
                st.subheader("Select type of Visualization")
                dfweight = pd.DataFrame.from_dict(proteinweight, orient="index",columns=["Number"])  # داتافريم خاصة وزن البروتين
                op=st.selectbox("",["Bar","Line","Area","Pie"])
                if op == "Bar":
                    st.bar_chart(dfweight)
                if op == "Line":
                    st.line_chart(dfweight)
                if op == "Area":
                    st.area_chart(dfweight)
                if op == "Pie":
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(proteinweight.values(), labels=proteinweight.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            # ********************************************************************************
            st.subheader("Possible Sequences")
            proteinseq1 = proteinseq1.replace("\n", '')
            proteinseq1 = proteinseq1.replace(' ', '')
            protein ={}
            for i in range(0, len(proteinseq1)): # باخد البروتينات اللي عندي في فاستا واشوف ايه السيكونس المتوقع ليها
                protein[proteinseq1[i]]=RNA_Codons_protientoseq[proteinseq1[i]]
            st.write(protein)
    if selected == "UniParc":
        UniParcEntry = st.text_area("Enter your FASTA of Protein from UniParc ", "Type Here")
        if st.checkbox("Submit UniParc"):
            st.balloons()
            with st.spinner('Wait for it...'):
                time.sleep(2)
            copy = UniParcEntry

            dictionary3 = {}
            UniParcEntry = UniParcEntry[1:]
            end = UniParcEntry.find("status=")
            dictionary3["Unique Identifier"] = [UniParcEntry[:end]]
            start = UniParcEntry.find("status=")
            end = UniParcEntry.find("\n")
            dictionary3["Status"] = [UniParcEntry[start + 7:end]]
            # ********************************************************************************
            df = pd.DataFrame.from_dict(dictionary3, orient="index", columns=["DATA"])# بحول الديكشنري للي عندي الي داتا فريم
            st.dataframe(df)# تعرضلي المعلومات بتاعت الفاستا بعد لما قسمتهم علي شكل داتا فريم
            # ********************************************************************************
            st.subheader("Protein Sequences")
            #s = UniParcEntry.find("status=")
            start = UniParcEntry.find("\n")
            proteinseq2 = UniParcEntry[start::1]
            st.success(proteinseq2)#بتعرض سيكونس البروتين الموجود في الفاستا
            # ********************************************************************************
            st.subheader("The Protein Weight")
            totalweight = ProtienWeight(proteinseq2)# فانكشن ببعتلها سيكونس البروتين بتحسبلي الويت وترجعه
            st.text(proteinweight)# البروتين ويت ده متغير نوعة ديكشناري مخزنه فيه الويت بتاعت البروتين
            st.success("The net weight = " + str(totalweight))
            # ********************************************************************************
            st.subheader("Number Of Protiens")
            d3 =Dictionary(proteinseq2)# بانده فانكشن وابعتلها البروتين بتعد عدد البروتينات وتجعهملي في ديكشناري
            dfcount = pd.DataFrame.from_dict(d3, orient="index", columns=["Number"])
            st.dataframe(dfcount)
            # ********************************************************************************
            st.header("Select Visualization")
            options = st.selectbox("", ["Visualization the Number of Protein", "Visualization of Protein Weight"])
            if options == "Visualization the Number of Protein":
                st.subheader("Select type of  Visualization")
                op = st.selectbox("", ["Bar", "Line", "Area", "Pie"])
                if op == "Bar":
                    st.bar_chart(dfcount)
                if op == "Line":
                    st.line_chart(dfcount)
                if op == "Area":
                    st.area_chart(dfcount)
                if op == "Pie":
                    newdict = {}  # ملت واحد جديد علشان اشيل القيم اللي بصفر علشان يكون الشكل مظبوط
                    for x in d3:
                        if d3[x] > 0:
                            newdict[x] = d3[x]
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(newdict.values(), labels=newdict.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            if options == "Visualization of Protein Weight":
                st.subheader("Select type of Visualization")
                dfweight = pd.DataFrame.from_dict(proteinweight, orient="index",columns=["Number"])  # داتافريم خاصة وزن البروتين
                op = st.selectbox("", ["Bar", "Line", "Area", "Pie"])
                if op == "Bar":
                    st.bar_chart(dfweight)
                if op == "Line":
                    st.line_chart(dfweight)
                if op == "Area":
                    st.area_chart(dfweight)
                if op == "Pie":
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(proteinweight.values(), labels=proteinweight.keys(), autopct='%1.1f%%', radius=1.5)
                    st.pyplot(fig)
            # ********************************************************************************
            st.subheader("Possible Sequences")
            proteinseq2 = proteinseq2.replace("\n", '')
            proteinseq2 = proteinseq2.replace(' ', '')
            protein ={}
            for i in range(0, len(proteinseq2)): # باخد البروتينات اللي عندي في فاستا واشوف ايه السيكونس المتوقع ليها
                 protein[proteinseq2[i]]=RNA_Codons_protientoseq[proteinseq2[i]]
            st.write(protein)
if selected == "DNA Analysis":
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_cftvyhwc.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json)
    st.header("FASTA Analysis")
    FASTA = st.text_area("Enter Your FASTA ", "Type Here..")#المكان اللي هندخل فيه الفاستا
    if st.checkbox("Analyze"):
        st.balloons()
        with st.spinner('Wait for it...'):
            time.sleep(1)
        dictionary = {}
        copy = FASTA  # عملت منه نسخه علشان لو عاوزة استخدمه بعد لما اعدل عليه
        # ********************************************************************************
        st.subheader("DNA Sequence")
        start = FASTA.find("\n")
        fasta = FASTA[start::1]
        st.success(fasta)
        # ********************************************************************************
        st.subheader("Protein Sequence")
        fasta = fasta.replace("\n", '')
        sequance = fasta.maketrans("ATGC", "UACG")
        RNA = fasta.translate(sequance)
        protein = ""
        for i in range(0, len(RNA), 3):
            codon = RNA[i:i + 3]
            if RNA_Codons_seqtoprotien[codon] != '*':#لو لقي ستوب كودون مش هيعرض باقي سيكونس
                protein += RNA_Codons_seqtoprotien[codon]
        st.write(protein)
        # ********************************************************************************
        st.subheader("Upload Fasta file for Alignment")
        uploaded_file = st.file_uploader("Upload Files", type=['txt','fasta','fsa'])
        if st.button("Process"):
            readseq=str(uploaded_file.read())
            start = readseq.find("\\n")
            readseq = readseq[start + 2:len(readseq) - 1:1]
            if uploaded_file is not None:
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,
                                "FileSize": uploaded_file.size}
                st.write(file_details)
            alignments = pairwise2.align.globalxx(fasta,readseq)
            for alignment in alignments:
                st.text(pairwise2.format_alignment(*alignment))
        # ********************************************************************************
            myseq = SeqIO.read(str(uploaded_file.name), "fasta")
            title = myseq.id
            seq =readseq
            win_size = 45
            i = 0
            number_l = []
            while i <= (len(seq) - win_size):
                number_l.append(seq[i:i + win_size].count("AAT"))
                i += 1
            pos = number_l.index(max(number_l))
            st.success(
                '''PRIMER_SEQUENCE_ID = %s
                \nSEQUENCE = %s
                \nTARGET = %s,%s
                \nPRIMER_OPT_SIZE = 18
                \nPRIMER_MIN_SIZE = 15
                \nPRIMER_MAX_SIZE = 20
                \nPRIMER_NUM_NS_ACCEPTED = 0
                \nPRIMER_EXPLAIN_FLAG = 1
                \nPRIMER_PRODUCT_SIZE_RANGE = %s-%s
                ''' % (title, seq, pos, win_size, win_size, len(seq)))
        # ********************************************************************************
        st.subheader("Number Of Nucleotides")
        d = dict([
            ('A', FASTA[start + 1::1].count('A')),
            ('T', FASTA[start + 1::1].count('T')),
            ('G', FASTA[start + 1::1].count('G')),
            ('C', FASTA[start + 1::1].count('C'))
        ])
        count = "A : " + str(d["A"]) + "  T : " + str(d["T"]) + "  C : " + str(d["C"]) + "  G : " + str(d["G"])
        st.success(count)
        # ********************************************************************************
        # داتافريم خاصة بعدد النيوكليوتيدات
        dfcount = pd.DataFrame.from_dict(d, orient="index", columns=["Nucleotides"])
        st.dataframe(dfcount)
        # ********************************************************************************
        st.subheader("Select type of Visualization")
        op = st.selectbox("", ["Bar", "Line", "Area", "Pie"])
        if op == "Bar":
            st.bar_chart(dfcount)
        if op == "Line":
            st.line_chart(dfcount)
        if op == "Area":
            st.area_chart(dfcount)
        if op == "Pie":
            fig = plt.figure(figsize=(5, 5))
            plt.pie(d.values(), labels=d.keys(), autopct='%1.1f%%')
            st.pyplot(fig)
if selected == "Classifying DNA With ML":
    lottie_url = "https://assets3.lottiefiles.com/packages/lf20_lm11cwm7.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json)
    st.subheader("Upload files")
    uploaded_file = st.file_uploader("Upload First file", type=['txt', 'fasta', 'fsa'])
    if st.checkbox("Process File1"):
        human_data = pd.read_table(uploaded_file.name) # class indicates that it is a family that it grows with according to the picture below
        st.write(human_data.head())
    uploaded_file2 = st.file_uploader("Upload Second file", type=['txt', 'fasta', 'fsa'])
    if st.checkbox("Process File2"):
        chimp_data = pd.read_table(uploaded_file2.name)
        st.write(chimp_data.head())
    uploaded_file3 = st.file_uploader("Upload Third file", type=['txt', 'fasta', 'fsa'])
    if st.checkbox("Process File3"):
        dog_data = pd.read_table(uploaded_file3.name)
        st.write(dog_data.head())
        st.subheader("They are gene sequence function groups.")
        st.image("Capture1.png")
        # lambda arguments: expression
        # Hold the sequence and apply a function getKmers to it then drop seq and replaces it with list "words"
        # بمعني هيستخدم lambda , يقسم ال seq ل 6 اجزاء
        # و بعدها يخزنها في ليست "word"
        human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        human_data = human_data.drop('sequence', axis=1)
        chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        chimp_data = chimp_data.drop('sequence', axis=1)
        dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        dog_data = dog_data.drop('sequence', axis=1)
        st.subheader("our coding sequence data is changed to lowercase, split up into all possible k - mer words of length 6 and ready for the next step.")
        st.write(human_data.head())

        st.write("**Since we are going to use scikit - learn natural language processing tools to do the k - mer counting, we need to now convert the lists of k - mers for each gene into string sentences of words that the count vectorizer can use.We can also make a y variable to hold the class labels.Let's do that now.**")
        # Take every list word "every whole row in all lists", put them next to each other
        # هياخد كل list word "كل صف بالكامل ب كل lists " , يحطها جنب بعض
        human_texts = list(human_data['words'])
        for item in range(len(human_texts)):
            human_texts[item] = ' '.join(human_texts[item])
        y_data = human_data.iloc[:, 0].values
        # لسه هفهم ال y_data
        st.write(human_texts[2])
        add="Y variable = "+str(y_data)
        st.text(add)

        chimp_texts = list(chimp_data['words'])
        for item in range(len(chimp_texts)):
            chimp_texts[item] = ' '.join(chimp_texts[item])
        y_chimp = chimp_data.iloc[:, 0].values  # y_c for chimp

        dog_texts = list(dog_data['words'])
        for item in range(len(dog_texts)):
            dog_texts[item] = ' '.join(dog_texts[item])
        y_dog = dog_data.iloc[:, 0].values

        # Creating the Bag of Words model using CountVectorizer()
        # This is equivalent to k-mer counting
        # The n-gram size of 4 was previously determined by testing

        cv = CountVectorizer(ngram_range=(4, 4))
        X = cv.fit_transform(human_texts)
        X_chimp = cv.transform(chimp_texts)
        X_dog = cv.transform(dog_texts)
        st.subheader("Now we will apply the BAG of WORDS using Count Vectorizer using NLP")
        st.success(X.shape)
        st.success(X_chimp.shape)
        st.success(X_dog.shape)
        st.subheader("If we have a look at class balance we can see we have relatively balanced dataset.")

        st.bar_chart(human_data['class'].value_counts().sort_index())

        # Splitting the human dataset into the training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)
        st.success(X_train.shape)
        st.success(X_test.shape)


        ### Multinomial Naive Bayes Classifier ###
        # The alpha parameter was determined by grid search previously

        classifier = MultinomialNB(alpha=0.1)
        st.write(classifier.fit(X_train, y_train))
        st.subheader("A multinomial naive Bayes classifier will be  created. I previously did some parameter tuning and found the ngram size of 4(reflected in the Countvectorizer() instance) and a model alpha of 0.1 did the best.")
        y_pred = classifier.predict(X_test)
        st.write("Confusion matrix\n")
        st.write(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))


        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        pri = "accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1)
        st.write(pri)

