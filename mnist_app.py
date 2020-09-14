import streamlit as st
import numpy as np 
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
#import censusdata
from PIL import Image, ImageOps

# for MNIST preprocessing 
from PIL import ImageFilter 
from matplotlib import pyplot as plt

#for AMLStudio JSON requests
import urllib.request
import json

#for plotting histograms
import numpy as np
import math 
from scipy import ndimage

# #for drawing
import cv2
# from streamlit_drawable_canvas import st_canvas

# function which will get the center of mass of a given image and returns the amount of shiftx and shifty needs to be done so that the image is centered. 
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

# This function shifts the image in a given direction using WarpAffine. 
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


st.write("Welcome to the Machine Learning Class! ")
st.write("Microsoft AED")
st.write("# MNIST - Handwriting Prediction ")
st.write("This is a simple image classification web app to predict handwritten numbers ")
st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("# Step 1")
st.write("Please copy the API key from the AML Studio webservice and paste here:")
apikeyforaml = st.text_input("API = ")
if len(apikeyforaml) is 0:
    st.text("You havent entered the API key")
    isapikey=0
else:
    isapikey=1



st.write("# Step 2")
st.write(' Goto your webservice (after deploying predictive experiment)' )
st.write( 'click "New Web Services Experience') 
st.write(' Under Basics, click "Use Endpoint" ') 
st.write(' Now copy the text in "Request-Response field' )
st.write(' Paste in the following dialog box.')

url = st.text_input("URL= ") #, sample_url)

st.write("# Step 3")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Step 1: Take the file and do file handling
if file is None:
    st.text("You haven't uploaded an image file")
    isimage=0
else:
    image = Image.open(file)                    #file variable should be of type image.png.  
    st.write("Fetching image...  ")
    st.write("Great! Your uploaded image is: ")
    st.image(image, use_column_width=True)
    isimage=1

# Step 2:  convert the Image into MNIST preprocessed image (28x28)
# OpenCV2 based preprocessing for MNIST with interpolation

if isimage == 1 and isapikey == 1:
    st.write("# step 4" )
    st.write("# preprocessing the image into MNIST format " )

    #read original image in cv2
    im_og = np.array(image)
    im_gray = cv2.cvtColor(im_og, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.resize(im_gray, (28, 28))                 #255-im_gray will invert the image to suit an handwritten digit in white background. 
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #removing rows and columns at the sides of the image which are completely black
    while np.sum(im_bw[0]) == 0:
        im_bw = im_bw[1:]

    while np.sum(im_bw[:,0]) == 0:
        im_bw = np.delete(im_bw,0,1)

    while np.sum(im_bw[-1]) == 0:
        im_bw = im_bw[:-1]

    while np.sum(im_bw[:,-1]) == 0:
        im_bw = np.delete(im_bw,-1,1)

    rows,cols = im_bw.shape      

    # rows and cols should be a 20x20 pixels so resize the image to 20x20 to fit the outer box.

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        im_bw = cv2.resize(im_bw, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        im_bw = cv2.resize(im_bw, (cols, rows))

    # In the end we need 28x28 pixel image. So adding black area again. 
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    im_bw = np.lib.pad(im_bw,(rowsPadding,colsPadding),'constant')
    
    #find the center of mass
    shiftx,shifty = getBestShift(im_bw)
    shifted = shift(im_bw,shiftx,shifty)
    im_bw = shifted
    
    
    st.image(im_bw, use_column_width=True)
    st.write(im_bw.shape[0])
    st.write(im_bw.shape[1])
 
    tv = im_bw.flatten() 
    st.write("After flattening:")
    st.write(len(tv))
    # tv = list(newImage.getdata())  # get pixel values
    
    # print("************ newRawImage without normalizing ********")
    # print(tv)

    ###### Enable to see Debug Logs ##############
    # st.write("Flatten the preprocessed image into: ", "1 x", len(tv))
    # st.write("Contents of the preprocessed image: ", tv)

    #data: 
    data = {
        "Inputs": {
                "input1":
                [
                    {
                            'Label': str(9),   
                            'f0': str(tv[0]),   
                            'f1': str(tv[1]),   
                            'f2': str(tv[2]),   
                            'f3': str(tv[3]),   
                            'f4': str(tv[4]),   
                            'f5': str(tv[5]),   
                            'f6': str(tv[6]),   
                            'f7': str(tv[7]),   
                            'f8': str(tv[8]),   
                            'f9': str(tv[9]),   
                            'f10': str(tv[10]),  
                            'f11': str(tv[11]),   
                            'f12': str(tv[12]),   
                            'f13': str(tv[13]),   
                            'f14': str(tv[14]),   
                            'f15': str(tv[15]),   
                            'f16': str(tv[16]),   
                            'f17': str(tv[17]),   
                            'f18': str(tv[18]),   
                            'f19': str(tv[19]),   
                            'f20': str(tv[20]),   
                            'f21': str(tv[21]),   
                            'f22': str(tv[22]),   
                            'f23': str(tv[23]),   
                            'f24': str(tv[24]),   
                            'f25': str(tv[25]),   
                            'f26': str(tv[26]),   
                            'f27': str(tv[27]),   
                            'f28': str(tv[28]),   
                            'f29': str(tv[29]),   
                            'f30': str(tv[30]),   
                            'f31': str(tv[31]),   
                            'f32': str(tv[32]),   
                            'f33': str(tv[33]),   
                            'f34': str(tv[34]),   
                            'f35': str(tv[35]),   
                            'f36': str(tv[36]),   
                            'f37': str(tv[37]),   
                            'f38': str(tv[38]),   
                            'f39': str(tv[39]),   
                            'f40': str(tv[40]),   
                            'f41': str(tv[41]),   
                            'f42': str(tv[42]),   
                            'f43': str(tv[43]),   
                            'f44': str(tv[44]),   
                            'f45': str(tv[45]),   
                            'f46': str(tv[46]),   
                            'f47': str(tv[47]),   
                            'f48': str(tv[48]),   
                            'f49': str(tv[49]),   
                            'f50': str(tv[50]),   
                            'f51': str(tv[51]),   
                            'f52': str(tv[52]),   
                            'f53': str(tv[53]),   
                            'f54': str(tv[54]),   
                            'f55': str(tv[55]),   
                            'f56': str(tv[56]),   
                            'f57': str(tv[57]),   
                            'f58': str(tv[58]),   
                            'f59': str(tv[59]),   
                            'f60': str(tv[60]),   
                            'f61': str(tv[61]),   
                            'f62': str(tv[62]),   
                            'f63': str(tv[63]),   
                            'f64': str(tv[64]),   
                            'f65': str(tv[65]),   
                            'f66': str(tv[66]),   
                            'f67': str(tv[67]),   
                            'f68': str(tv[68]),   
                            'f69': str(tv[69]),   
                            'f70': str(tv[70]),   
                            'f71': str(tv[71]),   
                            'f72': str(tv[72]),   
                            'f73': str(tv[73]),   
                            'f74': str(tv[74]),   
                            'f75': str(tv[75]),   
                            'f76': str(tv[76]),   
                            'f77': str(tv[77]),   
                            'f78': str(tv[78]),   
                            'f79': str(tv[79]),   
                            'f80': str(tv[80]),   
                            'f81': str(tv[81]),   
                            'f82': str(tv[82]),   
                            'f83': str(tv[83]),   
                            'f84': str(tv[84]),   
                            'f85': str(tv[85]),   
                            'f86': str(tv[86]),   
                            'f87': str(tv[87]),   
                            'f88': str(tv[88]),   
                            'f89': str(tv[89]),   
                            'f90': str(tv[90]),   
                            'f91': str(tv[91]),   
                            'f92': str(tv[92]),   
                            'f93': str(tv[93]),   
                            'f94': str(tv[94]),   
                            'f95': str(tv[95]),   
                            'f96': str(tv[96]),   
                            'f97': str(tv[97]),   
                            'f98': str(tv[98]),   
                            'f99': str(tv[99]),   
                            'f100': str(tv[100]),   
                            'f101': str(tv[101]),   
                            'f102': str(tv[102]),   
                            'f103': str(tv[103]),   
                            'f104': str(tv[104]),   
                            'f105': str(tv[105]),   
                            'f106': str(tv[106]),   
                            'f107': str(tv[107]),   
                            'f108': str(tv[108]),   
                            'f109': str(tv[109]),   
                            'f110': str(tv[110]),   
                            'f111': str(tv[111]),   
                            'f112': str(tv[112]),   
                            'f113': str(tv[113]),   
                            'f114': str(tv[114]),   
                            'f115': str(tv[115]),   
                            'f116': str(tv[116]),   
                            'f117': str(tv[117]),   
                            'f118': str(tv[118]),   
                            'f119': str(tv[119]),   
                            'f120': str(tv[120]),   
                            'f121': str(tv[121]),   
                            'f122': str(tv[122]),   
                            'f123': str(tv[123]),   
                            'f124': str(tv[124]),   
                            'f125': str(tv[125]),   
                            'f126': str(tv[126]),   
                            'f127': str(tv[127]),   
                            'f128': str(tv[128]),   
                            'f129': str(tv[129]),   
                            'f130': str(tv[130]),   
                            'f131': str(tv[131]),   
                            'f132': str(tv[132]),   
                            'f133': str(tv[133]),   
                            'f134': str(tv[134]),   
                            'f135': str(tv[135]),   
                            'f136': str(tv[136]),   
                            'f137': str(tv[137]),   
                            'f138': str(tv[138]),   
                            'f139': str(tv[139]),   
                            'f140': str(tv[140]),   
                            'f141': str(tv[141]),   
                            'f142': str(tv[142]),   
                            'f143': str(tv[143]),   
                            'f144': str(tv[144]),   
                            'f145': str(tv[145]),   
                            'f146': str(tv[146]),   
                            'f147': str(tv[147]),   
                            'f148': str(tv[148]),   
                            'f149': str(tv[149]),   
                            'f150': str(tv[150]),   
                            'f151': str(tv[151]),   
                            'f152': str(tv[152]),   
                            'f153': str(tv[153]),   
                            'f154': str(tv[154]),   
                            'f155': str(tv[155]),   
                            'f156': str(tv[156]),   
                            'f157': str(tv[157]),   
                            'f158': str(tv[158]),   
                            'f159': str(tv[159]),   
                            'f160': str(tv[160]),   
                            'f161': str(tv[161]),   
                            'f162': str(tv[162]),   
                            'f163': str(tv[163]),   
                            'f164': str(tv[164]),   
                            'f165': str(tv[165]),   
                            'f166': str(tv[166]),   
                            'f167': str(tv[167]),   
                            'f168': str(tv[168]),   
                            'f169': str(tv[169]),   
                            'f170': str(tv[170]),   
                            'f171': str(tv[171]),   
                            'f172': str(tv[172]),   
                            'f173': str(tv[173]),   
                            'f174': str(tv[174]),   
                            'f175': str(tv[175]),   
                            'f176': str(tv[176]),   
                            'f177': str(tv[177]),   
                            'f178': str(tv[178]),   
                            'f179': str(tv[179]),   
                            'f180': str(tv[180]),   
                            'f181': str(tv[181]),   
                            'f182': str(tv[182]),   
                            'f183': str(tv[183]),   
                            'f184': str(tv[184]),   
                            'f185': str(tv[185]),   
                            'f186': str(tv[186]),   
                            'f187': str(tv[187]),   
                            'f188': str(tv[188]),   
                            'f189': str(tv[189]),   
                            'f190': str(tv[190]),   
                            'f191': str(tv[191]),   
                            'f192': str(tv[192]),   
                            'f193': str(tv[193]),   
                            'f194': str(tv[194]),   
                            'f195': str(tv[195]),   
                            'f196': str(tv[196]),   
                            'f197': str(tv[197]),   
                            'f198': str(tv[198]),   
                            'f199': str(tv[199]),   
                            'f200': str(tv[200]),   
                            'f201': str(tv[201]),   
                            'f202': str(tv[202]),   
                            'f203': str(tv[203]),   
                            'f204': str(tv[204]),   
                            'f205': str(tv[205]),   
                            'f206': str(tv[206]),   
                            'f207': str(tv[207]),   
                            'f208': str(tv[208]),   
                            'f209': str(tv[209]),   
                            'f210': str(tv[210]),   
                            'f211': str(tv[211]),   
                            'f212': str(tv[212]),   
                            'f213': str(tv[213]),   
                            'f214': str(tv[214]),   
                            'f215': str(tv[215]),   
                            'f216': str(tv[216]),   
                            'f217': str(tv[217]),   
                            'f218': str(tv[218]),   
                            'f219': str(tv[219]),   
                            'f220': str(tv[220]),   
                            'f221': str(tv[221]),   
                            'f222': str(tv[222]),   
                            'f223': str(tv[223]),   
                            'f224': str(tv[224]),   
                            'f225': str(tv[225]),   
                            'f226': str(tv[226]),   
                            'f227': str(tv[227]),   
                            'f228': str(tv[228]),   
                            'f229': str(tv[229]),   
                            'f230': str(tv[230]),   
                            'f231': str(tv[231]),   
                            'f232': str(tv[232]),   
                            'f233': str(tv[233]),   
                            'f234': str(tv[234]),   
                            'f235': str(tv[235]),   
                            'f236': str(tv[236]),   
                            'f237': str(tv[237]),   
                            'f238': str(tv[238]),   
                            'f239': str(tv[239]),   
                            'f240': str(tv[240]),   
                            'f241': str(tv[241]),   
                            'f242': str(tv[242]),   
                            'f243': str(tv[243]),   
                            'f244': str(tv[244]),   
                            'f245': str(tv[245]),   
                            'f246': str(tv[246]),   
                            'f247': str(tv[247]),   
                            'f248': str(tv[248]),   
                            'f249': str(tv[249]),   
                            'f250': str(tv[250]),   
                            'f251': str(tv[251]),   
                            'f252': str(tv[252]),   
                            'f253': str(tv[253]),   
                            'f254': str(tv[254]),   
                            'f255': str(tv[255]),   
                            'f256': str(tv[256]),   
                            'f257': str(tv[257]),   
                            'f258': str(tv[258]),   
                            'f259': str(tv[259]),   
                            'f260': str(tv[260]),   
                            'f261': str(tv[261]),   
                            'f262': str(tv[262]),   
                            'f263': str(tv[263]),   
                            'f264': str(tv[264]),   
                            'f265': str(tv[265]),   
                            'f266': str(tv[266]),   
                            'f267': str(tv[267]),   
                            'f268': str(tv[268]),   
                            'f269': str(tv[269]),   
                            'f270': str(tv[270]),   
                            'f271': str(tv[271]),   
                            'f272': str(tv[272]),   
                            'f273': str(tv[273]),   
                            'f274': str(tv[274]),   
                            'f275': str(tv[275]),   
                            'f276': str(tv[276]),   
                            'f277': str(tv[277]),   
                            'f278': str(tv[278]),   
                            'f279': str(tv[279]),   
                            'f280': str(tv[280]),   
                            'f281': str(tv[281]),   
                            'f282': str(tv[282]),   
                            'f283': str(tv[283]),   
                            'f284': str(tv[284]),   
                            'f285': str(tv[285]),   
                            'f286': str(tv[286]),   
                            'f287': str(tv[287]),   
                            'f288': str(tv[288]),   
                            'f289': str(tv[289]),   
                            'f290': str(tv[290]),   
                            'f291': str(tv[291]),   
                            'f292': str(tv[292]),   
                            'f293': str(tv[293]),   
                            'f294': str(tv[294]),   
                            'f295': str(tv[295]),   
                            'f296': str(tv[296]),   
                            'f297': str(tv[297]),   
                            'f298': str(tv[298]),   
                            'f299': str(tv[299]),   
                            'f300': str(tv[300]),   
                            'f301': str(tv[301]),   
                            'f302': str(tv[302]),   
                            'f303': str(tv[303]),   
                            'f304': str(tv[304]),   
                            'f305': str(tv[305]),   
                            'f306': str(tv[306]),   
                            'f307': str(tv[307]),   
                            'f308': str(tv[308]),   
                            'f309': str(tv[309]),   
                            'f310': str(tv[310]),   
                            'f311': str(tv[311]),   
                            'f312': str(tv[312]),   
                            'f313': str(tv[313]),   
                            'f314': str(tv[314]),   
                            'f315': str(tv[315]),   
                            'f316': str(tv[316]),   
                            'f317': str(tv[317]),   
                            'f318': str(tv[318]),   
                            'f319': str(tv[319]),   
                            'f320': str(tv[320]),   
                            'f321': str(tv[321]),   
                            'f322': str(tv[322]),   
                            'f323': str(tv[323]),   
                            'f324': str(tv[324]),   
                            'f325': str(tv[325]),   
                            'f326': str(tv[326]),   
                            'f327': str(tv[327]),   
                            'f328': str(tv[328]),   
                            'f329': str(tv[329]),   
                            'f330': str(tv[330]),   
                            'f331': str(tv[331]),   
                            'f332': str(tv[332]),   
                            'f333': str(tv[333]),   
                            'f334': str(tv[334]),   
                            'f335': str(tv[335]),   
                            'f336': str(tv[336]),   
                            'f337': str(tv[337]),   
                            'f338': str(tv[338]),   
                            'f339': str(tv[339]),   
                            'f340': str(tv[340]),   
                            'f341': str(tv[341]),   
                            'f342': str(tv[342]),   
                            'f343': str(tv[343]),   
                            'f344': str(tv[344]),   
                            'f345': str(tv[345]),   
                            'f346': str(tv[346]),   
                            'f347': str(tv[347]),   
                            'f348': str(tv[348]),   
                            'f349': str(tv[349]),   
                            'f350': str(tv[350]),   
                            'f351': str(tv[351]),   
                            'f352': str(tv[352]),   
                            'f353': str(tv[353]),   
                            'f354': str(tv[354]),   
                            'f355': str(tv[355]),   
                            'f356': str(tv[356]),   
                            'f357': str(tv[357]),   
                            'f358': str(tv[358]),   
                            'f359': str(tv[359]),   
                            'f360': str(tv[360]),   
                            'f361': str(tv[361]),   
                            'f362': str(tv[362]),   
                            'f363': str(tv[363]),   
                            'f364': str(tv[364]),   
                            'f365': str(tv[365]),   
                            'f366': str(tv[366]),   
                            'f367': str(tv[367]),   
                            'f368': str(tv[368]),   
                            'f369': str(tv[369]),   
                            'f370': str(tv[370]),   
                            'f371': str(tv[371]),   
                            'f372': str(tv[372]),   
                            'f373': str(tv[373]),   
                            'f374': str(tv[374]),   
                            'f375': str(tv[375]),   
                            'f376': str(tv[376]),   
                            'f377': str(tv[377]),   
                            'f378': str(tv[378]),   
                            'f379': str(tv[379]),   
                            'f380': str(tv[380]),   
                            'f381': str(tv[381]),   
                            'f382': str(tv[382]),   
                            'f383': str(tv[383]),   
                            'f384': str(tv[384]),   
                            'f385': str(tv[385]),   
                            'f386': str(tv[386]),   
                            'f387': str(tv[387]),   
                            'f388': str(tv[388]),   
                            'f389': str(tv[389]),   
                            'f390': str(tv[390]),   
                            'f391': str(tv[391]),   
                            'f392': str(tv[392]),   
                            'f393': str(tv[393]),   
                            'f394': str(tv[394]),   
                            'f395': str(tv[395]),   
                            'f396': str(tv[396]),   
                            'f397': str(tv[397]),   
                            'f398': str(tv[398]),   
                            'f399': str(tv[399]),   
                            'f400': str(tv[400]),   
                            'f401': str(tv[401]),   
                            'f402': str(tv[402]),   
                            'f403': str(tv[403]),   
                            'f404': str(tv[404]),   
                            'f405': str(tv[405]),   
                            'f406': str(tv[406]),   
                            'f407': str(tv[407]),   
                            'f408': str(tv[408]),   
                            'f409': str(tv[409]),   
                            'f410': str(tv[410]),   
                            'f411': str(tv[411]),   
                            'f412': str(tv[412]),   
                            'f413': str(tv[413]),   
                            'f414': str(tv[414]),   
                            'f415': str(tv[415]),   
                            'f416': str(tv[416]),   
                            'f417': str(tv[417]),   
                            'f418': str(tv[418]),   
                            'f419': str(tv[419]),   
                            'f420': str(tv[420]),   
                            'f421': str(tv[421]),   
                            'f422': str(tv[422]),   
                            'f423': str(tv[423]),   
                            'f424': str(tv[424]),   
                            'f425': str(tv[425]),   
                            'f426': str(tv[426]),   
                            'f427': str(tv[427]),   
                            'f428': str(tv[428]),   
                            'f429': str(tv[429]),   
                            'f430': str(tv[430]),   
                            'f431': str(tv[431]),   
                            'f432': str(tv[432]),   
                            'f433': str(tv[433]),   
                            'f434': str(tv[434]),   
                            'f435': str(tv[435]),   
                            'f436': str(tv[436]),   
                            'f437': str(tv[437]),   
                            'f438': str(tv[438]),   
                            'f439': str(tv[439]),   
                            'f440': str(tv[440]),   
                            'f441': str(tv[441]),   
                            'f442': str(tv[442]),   
                            'f443': str(tv[443]),   
                            'f444': str(tv[444]),   
                            'f445': str(tv[445]),   
                            'f446': str(tv[446]),   
                            'f447': str(tv[447]),   
                            'f448': str(tv[448]),   
                            'f449': str(tv[449]),   
                            'f450': str(tv[450]),   
                            'f451': str(tv[451]),   
                            'f452': str(tv[452]),   
                            'f453': str(tv[453]),   
                            'f454': str(tv[454]),   
                            'f455': str(tv[455]),   
                            'f456': str(tv[456]),   
                            'f457': str(tv[457]),   
                            'f458': str(tv[458]),   
                            'f459': str(tv[459]),   
                            'f460': str(tv[460]),   
                            'f461': str(tv[461]),   
                            'f462': str(tv[462]),   
                            'f463': str(tv[463]),   
                            'f464': str(tv[464]),   
                            'f465': str(tv[465]),   
                            'f466': str(tv[466]),   
                            'f467': str(tv[467]),   
                            'f468': str(tv[468]),   
                            'f469': str(tv[469]),   
                            'f470': str(tv[470]),   
                            'f471': str(tv[471]),   
                            'f472': str(tv[472]),   
                            'f473': str(tv[473]),   
                            'f474': str(tv[474]),   
                            'f475': str(tv[475]),   
                            'f476': str(tv[476]),   
                            'f477': str(tv[477]),   
                            'f478': str(tv[478]),   
                            'f479': str(tv[479]),   
                            'f480': str(tv[480]),   
                            'f481': str(tv[481]),   
                            'f482': str(tv[482]),   
                            'f483': str(tv[483]),   
                            'f484': str(tv[484]),   
                            'f485': str(tv[485]),   
                            'f486': str(tv[486]),   
                            'f487': str(tv[487]),   
                            'f488': str(tv[488]),   
                            'f489': str(tv[489]),   
                            'f490': str(tv[490]),   
                            'f491': str(tv[491]),   
                            'f492': str(tv[492]),   
                            'f493': str(tv[493]),   
                            'f494': str(tv[494]),   
                            'f495': str(tv[495]),   
                            'f496': str(tv[496]),   
                            'f497': str(tv[497]),   
                            'f498': str(tv[498]),   
                            'f499': str(tv[499]),   
                            'f500': str(tv[500]),   
                            'f501': str(tv[501]),   
                            'f502': str(tv[502]),   
                            'f503': str(tv[503]),   
                            'f504': str(tv[504]),   
                            'f505': str(tv[505]),   
                            'f506': str(tv[506]),   
                            'f507': str(tv[507]),   
                            'f508': str(tv[508]),   
                            'f509': str(tv[509]),   
                            'f510': str(tv[510]),   
                            'f511': str(tv[511]),   
                            'f512': str(tv[512]),   
                            'f513': str(tv[513]),   
                            'f514': str(tv[514]),   
                            'f515': str(tv[515]),   
                            'f516': str(tv[516]),   
                            'f517': str(tv[517]),   
                            'f518': str(tv[518]),   
                            'f519': str(tv[519]),   
                            'f520': str(tv[520]),   
                            'f521': str(tv[521]),   
                            'f522': str(tv[522]),   
                            'f523': str(tv[523]),   
                            'f524': str(tv[524]),   
                            'f525': str(tv[525]),   
                            'f526': str(tv[526]),   
                            'f527': str(tv[527]),   
                            'f528': str(tv[528]),   
                            'f529': str(tv[529]),   
                            'f530': str(tv[530]),   
                            'f531': str(tv[531]),   
                            'f532': str(tv[532]),   
                            'f533': str(tv[533]),   
                            'f534': str(tv[534]),   
                            'f535': str(tv[535]),   
                            'f536': str(tv[536]),   
                            'f537': str(tv[537]),   
                            'f538': str(tv[538]),   
                            'f539': str(tv[539]),   
                            'f540': str(tv[540]),   
                            'f541': str(tv[541]),   
                            'f542': str(tv[542]),   
                            'f543': str(tv[543]),   
                            'f544': str(tv[544]),   
                            'f545': str(tv[545]),   
                            'f546': str(tv[546]),   
                            'f547': str(tv[547]),   
                            'f548': str(tv[548]),   
                            'f549': str(tv[549]),   
                            'f550': str(tv[550]),   
                            'f551': str(tv[551]),   
                            'f552': str(tv[552]),   
                            'f553': str(tv[553]),   
                            'f554': str(tv[554]),   
                            'f555': str(tv[555]),   
                            'f556': str(tv[556]),   
                            'f557': str(tv[557]),   
                            'f558': str(tv[558]),   
                            'f559': str(tv[559]),   
                            'f560': str(tv[560]),   
                            'f561': str(tv[561]),   
                            'f562': str(tv[562]),   
                            'f563': str(tv[563]),   
                            'f564': str(tv[564]),   
                            'f565': str(tv[565]),   
                            'f566': str(tv[566]),   
                            'f567': str(tv[567]),   
                            'f568': str(tv[568]),   
                            'f569': str(tv[569]),   
                            'f570': str(tv[570]),   
                            'f571': str(tv[571]),   
                            'f572': str(tv[572]),   
                            'f573': str(tv[573]),   
                            'f574': str(tv[574]),   
                            'f575': str(tv[575]),   
                            'f576': str(tv[576]),   
                            'f577': str(tv[577]),   
                            'f578': str(tv[578]),   
                            'f579': str(tv[579]),   
                            'f580': str(tv[580]),   
                            'f581': str(tv[581]),   
                            'f582': str(tv[582]),   
                            'f583': str(tv[583]),   
                            'f584': str(tv[584]),   
                            'f585': str(tv[585]),   
                            'f586': str(tv[586]),   
                            'f587': str(tv[587]),   
                            'f588': str(tv[588]),   
                            'f589': str(tv[589]),   
                            'f590': str(tv[590]),   
                            'f591': str(tv[591]),   
                            'f592': str(tv[592]),   
                            'f593': str(tv[593]),   
                            'f594': str(tv[594]),   
                            'f595': str(tv[595]),   
                            'f596': str(tv[596]),   
                            'f597': str(tv[597]),   
                            'f598': str(tv[598]),   
                            'f599': str(tv[599]),   
                            'f600': str(tv[600]),   
                            'f601': str(tv[601]),   
                            'f602': str(tv[602]),   
                            'f603': str(tv[603]),   
                            'f604': str(tv[604]),   
                            'f605': str(tv[605]),   
                            'f606': str(tv[606]),   
                            'f607': str(tv[607]),   
                            'f608': str(tv[608]),   
                            'f609': str(tv[609]),   
                            'f610': str(tv[610]),   
                            'f611': str(tv[611]),   
                            'f612': str(tv[612]),   
                            'f613': str(tv[613]),   
                            'f614': str(tv[614]),   
                            'f615': str(tv[615]),   
                            'f616': str(tv[616]),   
                            'f617': str(tv[617]),   
                            'f618': str(tv[618]),   
                            'f619': str(tv[619]),   
                            'f620': str(tv[620]),   
                            'f621': str(tv[621]),   
                            'f622': str(tv[622]),   
                            'f623': str(tv[623]),   
                            'f624': str(tv[624]),   
                            'f625': str(tv[625]),   
                            'f626': str(tv[626]),   
                            'f627': str(tv[627]),   
                            'f628': str(tv[628]),   
                            'f629': str(tv[629]),   
                            'f630': str(tv[630]),   
                            'f631': str(tv[631]),   
                            'f632': str(tv[632]),   
                            'f633': str(tv[633]),   
                            'f634': str(tv[634]),   
                            'f635': str(tv[635]),   
                            'f636': str(tv[636]),   
                            'f637': str(tv[637]),   
                            'f638': str(tv[638]),   
                            'f639': str(tv[639]),   
                            'f640': str(tv[640]),   
                            'f641': str(tv[641]),   
                            'f642': str(tv[642]),   
                            'f643': str(tv[643]),   
                            'f644': str(tv[644]),   
                            'f645': str(tv[645]),   
                            'f646': str(tv[646]),   
                            'f647': str(tv[647]),   
                            'f648': str(tv[648]),   
                            'f649': str(tv[649]),   
                            'f650': str(tv[650]),   
                            'f651': str(tv[651]),   
                            'f652': str(tv[652]),   
                            'f653': str(tv[653]),   
                            'f654': str(tv[654]),   
                            'f655': str(tv[655]),   
                            'f656': str(tv[656]),   
                            'f657': str(tv[657]),   
                            'f658': str(tv[658]),   
                            'f659': str(tv[659]),   
                            'f660': str(tv[660]),   
                            'f661': str(tv[661]),   
                            'f662': str(tv[662]),   
                            'f663': str(tv[663]),   
                            'f664': str(tv[664]),   
                            'f665': str(tv[665]),   
                            'f666': str(tv[666]),   
                            'f667': str(tv[667]),   
                            'f668': str(tv[668]),   
                            'f669': str(tv[669]),   
                            'f670': str(tv[670]),   
                            'f671': str(tv[671]),   
                            'f672': str(tv[672]),   
                            'f673': str(tv[673]),   
                            'f674': str(tv[674]),   
                            'f675': str(tv[675]),   
                            'f676': str(tv[676]),   
                            'f677': str(tv[677]),   
                            'f678': str(tv[678]),   
                            'f679': str(tv[679]),   
                            'f680': str(tv[680]),   
                            'f681': str(tv[681]),   
                            'f682': str(tv[682]),   
                            'f683': str(tv[683]),   
                            'f684': str(tv[684]),   
                            'f685': str(tv[685]),   
                            'f686': str(tv[686]),   
                            'f687': str(tv[687]),   
                            'f688': str(tv[688]),   
                            'f689': str(tv[689]),   
                            'f690': str(tv[690]),   
                            'f691': str(tv[691]),   
                            'f692': str(tv[692]),   
                            'f693': str(tv[693]),   
                            'f694': str(tv[694]),   
                            'f695': str(tv[695]),   
                            'f696': str(tv[696]),   
                            'f697': str(tv[697]),   
                            'f698': str(tv[698]),   
                            'f699': str(tv[699]),   
                            'f700': str(tv[700]),   
                            'f701': str(tv[701]),   
                            'f702': str(tv[702]),   
                            'f703': str(tv[703]),   
                            'f704': str(tv[704]),   
                            'f705': str(tv[705]),   
                            'f706': str(tv[706]),   
                            'f707': str(tv[707]),   
                            'f708': str(tv[708]),   
                            'f709': str(tv[709]),   
                            'f710': str(tv[710]),   
                            'f711': str(tv[711]),   
                            'f712': str(tv[712]),   
                            'f713': str(tv[713]),   
                            'f714': str(tv[714]),   
                            'f715': str(tv[715]),   
                            'f716': str(tv[716]),   
                            'f717': str(tv[717]),   
                            'f718': str(tv[718]),   
                            'f719': str(tv[719]),   
                            'f720': str(tv[720]),   
                            'f721': str(tv[721]),   
                            'f722': str(tv[722]),   
                            'f723': str(tv[723]),   
                            'f724': str(tv[724]),   
                            'f725': str(tv[725]),   
                            'f726': str(tv[726]),   
                            'f727': str(tv[727]),   
                            'f728': str(tv[728]),   
                            'f729': str(tv[729]),   
                            'f730': str(tv[730]),   
                            'f731': str(tv[731]),   
                            'f732': str(tv[732]),   
                            'f733': str(tv[733]),   
                            'f734': str(tv[734]),   
                            'f735': str(tv[735]),   
                            'f736': str(tv[736]),   
                            'f737': str(tv[737]),   
                            'f738': str(tv[738]),   
                            'f739': str(tv[739]),   
                            'f740': str(tv[740]),   
                            'f741': str(tv[741]),   
                            'f742': str(tv[742]),   
                            'f743': str(tv[743]),   
                            'f744': str(tv[744]),   
                            'f745': str(tv[745]),   
                            'f746': str(tv[746]),   
                            'f747': str(tv[747]),   
                            'f748': str(tv[748]),   
                            'f749': str(tv[749]),   
                            'f750': str(tv[750]),   
                            'f751': str(tv[751]),   
                            'f752': str(tv[752]),   
                            'f753': str(tv[753]),   
                            'f754': str(tv[754]),   
                            'f755': str(tv[755]),   
                            'f756': str(tv[756]),   
                            'f757': str(tv[757]),   
                            'f758': str(tv[758]),   
                            'f759': str(tv[759]),   
                            'f760': str(tv[760]),   
                            'f761': str(tv[761]),   
                            'f762': str(tv[762]),   
                            'f763': str(tv[763]),   
                            'f764': str(tv[764]),   
                            'f765': str(tv[765]),   
                            'f766': str(tv[766]),   
                            'f767': str(tv[767]),   
                            'f768': str(tv[768]),   
                            'f769': str(tv[769]),   
                            'f770': str(tv[770]),   
                            'f771': str(tv[771]),   
                            'f772': str(tv[772]),   
                            'f773': str(tv[773]),   
                            'f774': str(tv[774]),   
                            'f775': str(tv[775]),   
                            'f776': str(tv[776]),   
                            'f777': str(tv[777]),   
                            'f778': str(tv[778]),   
                            'f779': str(tv[779]),   
                            'f780': str(tv[780]),   
                            'f781': str(tv[781]),   
                            'f782': str(tv[782]),   
                            'f783': str(tv[783]),   
                    }
                ],
            },
        "GlobalParameters":  {}
    }


     # step3: send the image array to the WebAPI as described in the AML studio Classic [Prediction Experiment]

    st.write( "# Prediction")

    if st.button('Predict'):    
        st.write("Sending this array to Azure Machine Learning Studio ... ")
        body= str.encode(json.dumps(data))

        ####### Enable to see json that you are sending ##########
        #st.json(data)

        api_key =  apikeyforaml #'abc123' # Replace this with the API key for the web service
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
        req = urllib.request.Request(url, body, headers)
        try:
            response = urllib.request.urlopen(req)
            # response.read() returns a string response which you have to parse
            result = response.read()    
            # print(result)
            st.write("Received response from the AML Studio: ")

            ####### Enable to see result that you received as text ##########
            # st.write(result)              
            
            json_res= json.loads(result)    #converted string to json

            p0 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "0"'])
            p1 = float( json_res["Results"]["output1"][0]['Scored Probabilities for Class "1"'])
            p2 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "2"'])
            p3 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "3"'])
            p4 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "4"'])
            p5 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "5"'])
            p6 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "6"'])
            p7 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "7"'])
            p8 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "8"'])
            p9 = float(json_res["Results"]["output1"][0]['Scored Probabilities for Class "9"'])
            
            probs = [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]

            max_prob = max(probs)
            indices = [i for i,j in enumerate(probs) if j==max_prob]
            
            st.write("Probability of class 0 :", p0)
            st.write("Probability of class 1 :", p1)
            st.write("Probability of class 2 :", p2)
            st.write("Probability of class 3 :", p3)
            st.write("Probability of class 4 :", p4)
            st.write("Probability of class 5 :", p5)
            st.write("Probability of class 6 :", p6)
            st.write("Probability of class 7 :", p7)
            st.write("Probability of class 8 :", p8)
            st.write("Probability of class 9 :", p9)        
            st.write("Max probability: ", max_prob)
            
            if len(indices) == 1:
                st.write(f' # Prediction: {indices[0]}')
            else:
                st.write("# prediction:")
                for i in indices:
                    st.write(i)

        except urllib.error.HTTPError as error:
            st.write("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            st.write("Error Info: ")
            st.write(error.info())
            st.write("************")
            st.write(json.loads(error.read().decode("utf8", 'ignore'))) 

