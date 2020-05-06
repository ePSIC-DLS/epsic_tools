def FuncAberrUV(u,v,aberrcoeff):
    
    # this function needs comments
     
    # input:
    # u: Kx
    # v: Ky

    u2 = u*u
    u3 = u2*u
    u4 = u3*u
    
    v2 = v*v
    v3 = v2*v
    v4 = v3*v
    
    # aberr are in unit of meter.
    C1   = aberrcoeff[0] # defocus
    C12a = aberrcoeff[1] # 2 stig
    C12b = aberrcoeff[2] # 2 stig
    C23a = aberrcoeff[3] # 3 stig
    C23b = aberrcoeff[4] # 3 stig
    C21a = aberrcoeff[5] # coma 
    C21b = aberrcoeff[6] # coma
    C3   = aberrcoeff[7] # Spherical abb
    C34a = aberrcoeff[8] # 4 stig
    C34b = aberrcoeff[9] # 4 stig
    C32a = aberrcoeff[10] # star
    C32b = aberrcoeff[11] # star
    
    # output:  chi function. in unit of meter*radian.  multiply by 2pi/lambda to get dimensionless
    func_aberr =  1/2*C1*(u2+v2)\
            + 1/2*(C12a*(u2-v2) + 2*C12b*u*v)\
            + 1/3*(C23a*(u3-3*u*v2) + C23b*(3*u2*v - v3))\
            + 1/3*(C21a*(u3+u*v2) + C21b*(v3+u2*v))\
            + 1/4* C3*(u4+v4+2*u2*v2)\
            + 1/4* C34a*(u4-6*u2*v2+v4)\
            + 1/4* C34b*(4*u3*v-4*u*v3)\
            + 1/4* C32a*(u4-v4)\
            + 1/4* C32b*(2*u3*v + 2*u*v3)\
    
    return func_aberr

