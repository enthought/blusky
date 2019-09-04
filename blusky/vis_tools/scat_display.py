
import functools
import matplotlib.pyplot as plt
import numpy as np

class nAttrDict(dict):
    """
    A class to convert a nested Dictionary into an object with key-values
    that are accessible using attribute notation (AttrDict.attribute) instead of
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: AttrDict.attr.attr)
    """
    def __init__(self, iterable, **kwargs):
        super(nAttrDict, self).__init__(iterable, **kwargs)
        for key, value in iterable.items():
            if isinstance(value, dict):
                self.__dict__[key] = nAttrDict(value)
            else:
                self.__dict__[key] = value  

class AttrDict(dict):   
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def getoptions(options, name, default):
    try:
        options.name
        return getattr(options, name)
    except KeyError:
        return default

def simplestack(code, base, len_):
    out = 0
    for l in range(0,len_):
        if len_ > 1: out = base*out + code[l]
        else: out = base*out + code # this will not work if code is an array
    return out

def recover_meta(inner,dirac):
    J=inner.m0.meta.j;
    L=np.max(inner.m1.meta.theta);
    
    meta = AttrDict()
    meta.order=1
    meta.scale=-1
    meta.orientation=0
    meta.dirac_norm=np.linalg.norm(dirac.m0.signal)
    meta.ave=np.mean(inner.m0.signal)
    r=2-1
    for m in range(2,len(inner.keys())+1):
        inner_m = nAttrDict(inner['m'+str(m-1)])
        dirac_m = nAttrDict(dirac['m'+str(m-1)])
        for l in range(0,len(inner_m.signal)): # might need to change the structure of signal
            meta.order = np.append(meta.order,m);
            # meta.scale.r=simplestack(inner['m'+str(m-1)].meta.j[1:-1],l,J,m-1);
            # meta.orientation.r=simplestack(inner['m'+str(m-1)].meta.theta[:,l]-1,L,m-1);
            # meta.dirac_np.norm.r=np.norm(dirac['m'+str(m-1)].signal)
            meta.scale = np.append(meta.scale,simplestack(inner_m.meta.j[0:-1,l],J,m-1))
            if m==2: meta.orientation = np.append(meta.orientation,simplestack(inner_m.meta.theta[l]-1,L,m-1))
            else: meta.orientation = np.append(meta.orientation,simplestack(inner_m.meta.theta[:,l]-1,L,m-1))
            meta.dirac_norm = np.append(meta.dirac_norm,np.linalg.norm(dirac_m.signal[l]))
            meta.ave = np.append(meta.ave,np.mean(inner_m.signal[l]))
            r=r+1
            
    return meta
 #How to import Meta definintion from mmatlab... send sda version?     
 
def effective_energy(meta,use_whole_ener):
    R=len(meta.order);
    maxorder=max(meta.order);
    first_mask = np.where(meta.order==2)
    J=max(meta.scale[first_mask])+1;
    L=max(meta.orientation[first_mask])+1;
    
    # last_mask=str.find(meta.order==maxorder);
    last_mask = np.where(meta.order==maxorder);
    metaout=meta;
    metaout.dirac_effnorms = np.zeros([np.size(meta.dirac_norm)])
    metaout.dirac_effnorms[last_mask]=meta.dirac_norm[last_mask]**2
    if use_whole_ener:
        metaout.dirac_effnorms[last_mask]=meta.dirac_onorm[last_mask]**2;
  
    for o in np.arange(maxorder-1,0,-1):
        slice_ = np.where(meta.order==o)[0]
        for s in slice_:
        #find children
            if o == 1:
                children=np.where(meta.order==o+1)[0]
            else:
                children=np.where((np.floor(meta.scale/J)==meta.scale[s])&(np.floor(meta.orientation/L)==meta.orientation[s])&(meta.order==o+1))[0]
            metaout.dirac_effnorms[s]=sum(metaout.dirac_effnorms[children])+meta.dirac_norm[s]**2
        if use_whole_ener:
            metaout.dirac_effnorms[s]=metaout.dirac_onorm[s]**2;

    return metaout
    
def split_rectangle(inrectangle, scales, orientations, dirac_phi, dirac_orig, dirac_norms, J, L, use_whole_ener, unified_plot, logpolar,order):
#first step: we marginalize orientations in order to split scale axis:
    C=len(dirac_norms);
    if unified_plot or order==1:
        ener=dirac_phi**2;
        totener=sum(dirac_norms)+ener;
    else: 
        totener=sum(dirac_norms);
    if use_whole_ener:
        totener=dirac_orig**2;
    totalheight = inrectangle[1]-inrectangle[0]
    if logpolar:
        totalwidth=inrectangle[-1]**2-inrectangle[-2]**2;
    else:
        totalwidth=inrectangle[-1]-inrectangle[-2]
    
    outlowp=inrectangle*1.0
    if unified_plot or order==1:
        if logpolar:
            outlowp[-1]= np.sqrt(outlowp[-2]**2+totalwidth*(ener/totener))
        else:
            outlowp[-1] = outlowp[-2]+totalwidth*(ener/totener);
        rasterwidth=outlowp[-1];
    else:
        rasterwidth=inrectangle[2]
    scale_parent= np.mod(np.floor(scales/J),J);
    orient_parent=np.mod(np.floor(orientations/L),L);
    diffori=1

    out = np.zeros([len(scales),4])
    for j in np.arange(J-1,-1,-1):
        pack=np.where(np.mod(scales,J)==j)[0]
        if pack.size != 0:#not isempty(pack):
            width=sum(dirac_norms[pack]**1);
            rasterheight=inrectangle[0]
            for l in np.arange(0,L,1):
                ind=np.where(np.mod(orientations[pack]-diffori*orient_parent[pack]+(order>1)*L/2,L)==l)[0]
                out[pack[ind],0]=rasterheight;
                out[pack[ind],1]=rasterheight+totalheight*dirac_norms[pack[ind]]**1/width
                out[pack[ind],2]=rasterwidth;
                if logpolar:
                    out[pack[ind],3]=np.sqrt(rasterwidth**2+totalwidth*width/totener)
                else:
                    out[pack[ind],3]=rasterwidth+totalwidth*width/totener;
                rasterheight=out[pack[ind],1]
            # print(order,j,rasterwidth)
            # if order==2: pdb.set_trace()
            if logpolar:
                rasterwidth=np.sqrt(rasterwidth**2+totalwidth*width/totener)
            else:
                rasterwidth=rasterwidth+totalwidth*width/totener
                
    if order==3 and not use_whole_ener:
        in_area = totalheight*totalwidth;
        if unified_plot:
            if logpolar:
                out_area = (outlowp[1]-outlowp[0])*(outlowp[3]**2-outlowp[2]**2) + sum((out[:,1]-out[:,0])**(out[:,3]**2-out[:,2]**2));
            else:
                out_area = (outlowp[1]-outlowp[0])*(outlowp[3]-outlowp[2]) + sum((out[:,1]-out[:,0])*(out[:,3]-out[:,2])); 
        else:
            if logpolar:
                out_area =  sum((out[:,1]-out[:,0])*(out[:,3]**2-out[:,2]**2));
            else:
                out_area =  sum((out[:,1]-out[:,0])*(out[:,3]-out[:,2]));
    
        tol=1e-5;
        if abs(in_area-out_area) > tol*in_area: #in_area and out_area? is that a function or variable?
            print('sthg weird')
            
    return out, outlowp 
    
def compute_rectangles(meta,use_whole_ener,unified_plot, logpolar):
    LP_correction = 1
    R = len(meta.order)
    maxorder = max(meta.order)
    first_mask = np.where(meta.order==2)[0]
    J = max(meta.scale[first_mask])+1
    L = max(meta.orientation[first_mask])+1


    meta.lp_correction = 1 #min(1,meta.lp_correction)
    meta.rectangle = np.array([[0,1,0, np.sqrt(meta.lp_correction)]])
    #meta.dirac_onorm(1)=meta.dirac_onorm(1)*LP_correction
    meta.dirac_onorm = meta.dirac_norm
    
    for o in range(1,maxorder+1):
        slice_ = np.where(meta.order==o)[0]
        for s in slice_:
    #find children
            if o == 1:
                children = np.where(meta.order==o+1)[0]
            else:
                #
                children = np.where((np.floor(meta.scale/J)==meta.scale[s])&(np.floor(meta.orientation/L)==meta.orientation[s])&(meta.order==o+1))[0]
            if children.size!=0:
                newrectangles, outrect = split_rectangle(meta.rectangle[s],meta.scale[children], \
                    meta.orientation[children],meta.dirac_norm[s],meta.dirac_onorm[s], \
                    meta.dirac_effnorms[children],J,L,use_whole_ener,unified_plot,logpolar,o)
                for c in range(0,len(children)):
                    meta.rectangle = np.insert(meta.rectangle,[len(meta.rectangle)],newrectangles[c,:],axis=0)
                    #try:
                    #    meta.rectangle = np.insert(meta.rectangle,[len(meta.rectangle)],newrectangles[c,:],axis=0)
                    #except IndexError:
                    #    pdb.set_trace()
                meta.covered[s] = ((outrect[1]-outrect[0])*(outrect[3]-outrect[2])+ \
                    sum((newrectangles[:,1]-newrectangles[:,0])**(newrectangles[:,3]-newrectangles[:,2])))/ \
                    ((meta.rectangle[s,1]-meta.rectangle[s,0])*(meta.rectangle[s,3]-meta.rectangle[s,2]))
                meta.rectangle[s,:]=outrect;

    # pdb.set_trace()
    return meta
 
def logpolar_conversion(inn,L):
    
    N,M = np.shape(inn)

    ix=np.array([np.arange(-M,M+1)])
    iix = np.matmul(np.ones([len(ix[0]),1]),ix)
    
    iiy=np.transpose(iix*1.0)
    r=np.sqrt(iix**2+iiy**2)
    theta=np.angle(iiy+1.0j*iix)
    N2 = len(r)
    M2 = len(r[0,:])
    # [N2,M2]=np.size(r);
    #r=r(:,round(M2/2):end);
    mask=(r>M);
    #theta=theta(:,round(M2/2):end);
    theta=np.mod(theta+(0.0*L+1.0)*np.pi/(2.0*L),np.pi)
    theta=theta/np.pi;
    
    code=np.minimum(N-2,np.maximum(1,np.round(theta*N)))+N*np.minimum(M-1,np.maximum(1,np.round(r)))
    ind_ = np.arange(1,512**2+1)
    ind_ = ind_.reshape((N,M))
    inn_ind = np.maximum(1,np.minimum(inn.size,1+code)) - 1
    inn_ind = inn_ind.astype(int)
    # row_ind = np.zeros([N2,M2])
    # col_ind = np.zeros([N2,M2])
    # pdb.set_trace()
    # for ii in range(0,N2):
    #     for jj in range(0,M2):
    #         row_ind[ii,jj] = np.where(ind_==inn_ind[ii,jj])[0]
    #         col_ind[ii,jj] = np.where(ind_==inn_ind[ii,jj])[1]
    #         print(ii,jj)
    inn_ind = inn_ind.reshape((inn_ind.size,1), order='F')
    inn_ = inn*1.0
    inn_ = inn_.reshape((inn.size,1), order='F')
    out=inn_[inn_ind]
    # out = np.transpose(out.reshape((N2,M2), order='C'))
    out = out.reshape((N2,M2), order='F')
    out[mask]=0
    # plt.figure(np.round(np.random.rand()*10)), plt.imshow(out,cmap='jet'), plt.colorbar(), plt.show(), plt.pause(1)
    # pdb.set_trace()
    del theta
    theta = np.zeros([N2,M2,2])
    theta[:,:,0]=np.mod(code+1,N)
    theta[:,:,1]=np.minimum(M-1,np.maximum(1,np.round(r)))
    
    return out,theta
    # return out
 

def scat_display(inner,dirac,options,scatt):
# def scat_display(in_obj=0, dirac=1, options=2, scatt=3):
    # imout = in_obj
    # orderout = dirac
    # cimout = options
    # meta = scatt
    # thetaout = 4
    
    options = AttrDict(options)        
    options.null = 0
    #print('null is ' + str(options.null)) #why?
    
    use_whole_ener=getoptions(options,'use_whole_ener',0) 
    null=getoptions(options,'null',-999)
    type_=getoptions(options,'display_type',2)
    logpolar=getoptions(options,'display_logpolar',1)
    renorm_process=getoptions(options,'renorm_process',0)
    maxsize=getoptions(options,'display_type',512)
    
    J=inner.m0.meta.j
    L=np.max(inner.m1.meta.theta)
      
    meta = recover_meta(inner,dirac)  
    # if nargin < 4: 
    #     scatt=meta.ave
    try: scatt
    except NameError: scatt=meta.ave
    meta.covered=np.zeros(np.size(meta.order))
    meta=effective_energy(meta,use_whole_ener)
    meta=compute_rectangles(meta,use_whole_ener,type_==0,logpolar) 
    maxorder=np.max(meta.order)
    l2_renorm=0
    if renorm_process:
        norm_ratio = scatt.meta.dirac_norm*1.0
    else:
        l2_renorm=getoptions(options,'l2_renorm',0)
        denom=np.ones(np.size(meta.dirac_norm))
        if l2_renorm:
            denom=np.sqrt(2**(-np.mod(meta.scale,J)))
        
        if (np.size(scatt)!=np.size(denom)):
            scatt = np.transpose(scatt)

        # If only single grid
        if np.shape(scatt)[1]==1:
            scatt = scatt.reshape(len(scatt))
        norm_ratio=scatt/denom 
    
    
    heights = meta.rectangle [:], 2 - meta.rectangle [:], 1; 
    widths = meta.rectangle [:], 4 - meta.rectangle [:], 3;
    
    fact_h = maxsize;
    fact_w = maxsize;
    
    imout = {}
    if type_ < 2:
        imout=np.zeros((fact_h+1,fact_w+1));
        orderout=np.np.zeros(fact_h+1,fact_w+1);
    else:
        for m in range(1, maxorder+1):
            imout[str(m)]=np.zeros(([fact_h,fact_w]));
    
    # # NOT CURRENTLY WORKING FOR TYPE=0, NOR TYPE=1   
    if type_ == 1:
        for Ord in 1, maxorder:
            selected=str.find(meta.order==ord);
            ordener(Ord).lvalue=sum(meta.dirac_norm(selected)**2);
        #end
        ordener=ordener/sum(ordener);
        cordener=np.cumsum(ordener);
        cordener = [0, cordener]; 
        #concatenate and squeeze the rectangles
        lower=0;
        for Ord in 1, maxorder:
            selected=str.find(meta.order==ord);
            if ordener(Ord)>0:
                if logpolar:
                    #obtain the upper and lower bounds of the annulus
                    upper = np.sqrt(lower ** 2 + ordener(Ord))
                    meta.rectangle[selected, 3:4] = np.sqrt(meta.rectangle[selected,3:4]**2*upper**2+(1 - meta.rectangle[selected,3:4] ** 2) * lower ** 2)
                    lower = upper;
                else:
                    meta.rectangle[selected,3:4]=ordener(Ord)*meta.rectangle[selected,3:4]+cordener(Ord);
        type_=0
    
    if type_ == 0:
        for l in 1:
            len(norm_ratio)
        #extrema
        ext[1] = 1 + np.floor(fact_h * meta.rectangle(l, 1))
        ext[2] = 1 + np.floor(fact_h * meta.rectangle(l, 2))
        ext[3] = 1 + np.floor(fact_w * meta.rectangle(l, 3))
        ext[4] = 1 + np.floor(fact_w * meta.rectangle[l, 4])
    
        inthh = (ext[1],ext[2]);
        intww = (ext[3],ext[4])
        imout[inthh, intww] = norm_ratio(l)
        orderout[inthh, intww] = meta.order(l);
    
        if logpolar:
            [imout, thetaout] = logpolar_conversion(imout, L)
            [orderout] = logpolar_conversion(orderout, L)
        
            m1 = (orderout == 2)
            m2 = (orderout == 3)
            m3 = (orderout > 3)
            [gox, goy] = gradient(orderout)
            couronnes=(conv2(gox**goy**np.ones[5], 'same') <.25)
            nimout=imout/max(imout[:])
            nimout=nimout**couronnes+(1-couronnes)
            [NN, MM] = np.size(nimout)
            cimout = np.ones(NN, MM, 3)
            cimout[:,:,1]=1-nimout**(m2|m3);
            cimout[:,:,3]=1-nimout**(m1|m3);
            cimout[:,:,2]=1-nimout**(m1|m2);    
        
    elif type_ == 2:
        # THERE IS A DISCREPENCY WITH MATLAB AND PYTHON BECAUSE OF PRECISION THAT PRESENTS ITSELF IN NP.FLOOR BELOW
        meta.rectangle[np.where(np.abs(meta.rectangle-1)<1.0e-5)] = 1.0
        meta.rectangle[np.where(np.abs(meta.rectangle-0.5)<1.0e-5)] = 0.5
        for Ord in range(2,maxorder+1):
            selected = np.where(meta.order == Ord)[0]
            for l in selected:
                inth = np.minimum(fact_h, [1 + np.floor(fact_h * meta.rectangle[l, 0]), np.floor(fact_h * meta.rectangle[l, 1])])
                # inth = np.minimum(fact_h, [1 + np.round(fact_h * meta.rectangle[l, 0]), np.around(fact_h * meta.rectangle[l, 1],decimals=14)])
                inth = inth.astype(int)-1
                intw = np.minimum(fact_w, [1 + np.floor(fact_w * meta.rectangle[l, 2]), np.floor(fact_w * meta.rectangle[l, 3])])
                # intw = np.minimum(fact_w, [1 + np.round(fact_w * meta.rectangle[l, 2]), np.round(fact_w * meta.rectangle[l, 3])])
                intw = intw.astype(int)-1
                imout[str(Ord-1)][inth.min():inth.max()+1,intw.min():intw.max()+1] = norm_ratio[l]
                # print(l,imout[str(Ord-1)].sum())
                # if l==4: pdb.set_trace()
    
            if logpolar:
                # plt.figure(Ord*2), plt.imshow(imout[str(Ord - 1)],cmap='jet'), plt.colorbar(), plt.show()
                imout[str(Ord - 1)], thetaout = logpolar_conversion(imout[str(Ord - 1)], L)
                # plt.figure(Ord*6), plt.imshow(imout[str(Ord - 1)],cmap='viridis'), plt.colorbar(), plt.show()
    
    # pdb.set_trace()
    # try: return imout, orderout, cimout, meta, thetaout
    # except UnboundLocalError: return imout, cimout, meta, thetaout
    return imout

    

    
