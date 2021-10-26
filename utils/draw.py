
def show_layers(structure):
    n = len(structure)
    layers = ""

    for i in range(n):
        if i!=0:
            layers+='\n'
        keyword = list(structure[i].keys())[0]
        if keyword=='conv':
            c,k_x,k_y = list(structure[i].values())[0]
            h-=k_x
            w-=k_y
            layers += f"layer({i}):conv | {c}x{h}x{w}"
        elif keyword=='pool':
            kernel_size = list(structure[i].values())[0]
            h-=kernel_size
            w-=kernel_size
            layers += f"layer({i}):pool | {c}x{h}x{w}"
        elif keyword=='dropout':
            p = list(structure[i].values())[0]
            layers +=f"layer({i}):dropout | p = {p}"
        elif keyword  == 'activation':
            type  = list(structure[i].values())[0]
            layers +=f"layer({i}):activation | type = {type}"
        elif keyword =='linear':
            IN ,OUT = list(structure[i].values())[0]
            layers +=f"layer({i}):linear | input = {IN} output = {OUT}"
        elif keyword=='orgin':
            c,h,w = list(structure[i].values())[0]
            layers += f"layer({i}):origin | {c}x{h}x{w}"
        elif keyword =="change_dim":
            c = list(structure[i].values())[0]
            layers += f"layer({i}):change_dim | {c}"
        else :
            raise KeyError
    return layers



# def plot_layer(ax,dx, dy, dz,x,y,z):
#     x=x
#     y=y
#     z=z

#     xx = np.linspace(x, dx,2)#等差数列
#     yy = np.linspace(y, dy,2)
#     zz = np.linspace(z, dz,2)

#     xx2, yy2 = np.meshgrid(xx, yy)

#     ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
#     ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
   

#     yy2, zz2 = np.meshgrid(yy, zz)
#     ax.plot_surface(np.full_like(yy2, x), yy2, zz2)
#     ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)

#     xx2, zz2= np.meshgrid(xx, zz)
#     ax.plot_surface(xx2, np.full_like(yy2, y), zz2)
#     ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)

#     axisEqual3D(ax)
#     ax.view_init(elev=20., azim=-60)#这个角度比较合适
#     plt.axis([-20,20,-50,50])
#     plt.axis('off')  #去掉坐标轴


# def axisEqual3D(ax):#xyz单位长度统一
#     extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
#     sz = extents[:,1] - extents[:,0]
#     centers = np.mean(extents, axis=1)
#     maxsize = max(abs(sz))
#     r = maxsize/2
#     for ctr, dim in zip(centers, 'xyz'):
#         getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


    

