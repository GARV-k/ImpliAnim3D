import torch
import argparse
import tinycudann as tcnn
from manage_objs import *
from sdf.utils2 import *
from hash_imports.encodings import HashEncoding
from sdf.meshsdf_decoder import MeshSDFDecoder

def save_mesh(models, N=None,M=None,save_path='meshes_using_hashencoder/', resolution=256):
        for j, model in enumerate(models):
            if N == None or M == None:
                save_path1 = save_path+f'0.obj'
            else: 
                save_path1 = save_path+f'N_M_0.obj'
            print(f"==> Saving mesh to {save_path1}")

            os.makedirs(os.path.dirname(save_path1), exist_ok=True)

            def query_func(pts,model):
                pts = pts.to('cuda')
                with torch.no_grad():   
                    with torch.cuda.amp.autocast(enabled=True):
                        # model.encoder.params = torch.nn.Parameter(model.encoder.params*w.params)
                        sdfs = model(pts,N,M) #TODO
                return sdfs

            bounds_min = torch.FloatTensor([-1, -1, -1])
            bounds_max = torch.FloatTensor([1, 1, 1])
            
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func, model=model)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(save_path1)
        # vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
            print(f"==> Finished saving mesh {idx}/{len(models)-1}.")
        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(save_path)

        print(f"==> Finished saving meshes.")



                 
                 

if __name__ == '__main__':
    from pipeline_args import pipeline_parser, parse
    parser = pipeline_parser("Shared-decoder training: one hash table per mesh (unrotated and rotated), one decoder for all.", epochs=1000)
    parser.add_argument("--freeze-decoder", action="store_true", help="only train the hash tables, keep the decoder at its initialisation")
    args = parse(parser)

    n = args.n
    d = args.d
    hashmap_size = args.hashmap_size
    obj_no = args.obj_no

    obj_path_pre = args.data_root

    obj_path_pre_list = [
                            obj_path_pre+'/'
                            ,obj_path_pre+'_90/'
                            # ,obj_path_pre+'_180/'
                            # ,obj_path_pre+'_270/'
                        ]
    workspace1 = args.workspace
    # workspace1 = f'Exp12/workspace/'
    # workspace2 = 'hash_workspace_obj12_7_8_10/'
    _n = args.num_objs
    # idx_list = [360,180]
    obj_list = [obj_path+f"{idx+1}.obj" for idx in range(_n) for obj_path in obj_path_pre_list]
    _n = len(obj_list)
    print(f"total objs: {_n}")
    
    from sdf.network_tcnn import SDFNetwork   
             
    # tcnn_network = tcnn.Network(
    #                                     n_input_dims=32,
    #                                     n_output_dims=1,
    #                                     network_config={
    #                                         "otype": "FullyFusedMLP",
    #                                         "activation": "ReLU",
    #                                         "output_activation": "None",
    #                                         "n_neurons": 64,
    #                                         "n_hidden_layers": 2,
    #                                     },
    #                                 )
    tcnn_network = MeshSDFDecoder(num_layers=8,
                 skips=[],
                 hidden_dim=128,
                 clip_sdf=None,
                 in_dim = 32,
                 out_dim = 1,)
    
    # tcnn_network = torch.load(workspace2+'GT_mlp.pth')
    # tcnn_network.load_state_dict(mlp_state_dict)
    # tcnn_network = torch.load(workspace1+'GT_mlp.pth')
    
    models = []
    for idx in range(_n):
        model = SDFNetwork( tcnn_network, encoding="hashgrid",hashmap_size=hashmap_size)
        # enc_state_dict = torch.load(f'saved_models/enc_{idx}.pth')
        # model.encoder.load_state_dict(enc_state_dict)
        # model.encoder = torch.load(workspace1+f'GT_enc_{idx}.pth')
        models.append(model)
    # model2 = SDFNetwork(tcnn_network, encoding="hashgrid") 
    # models = [model, model2]
    # print(models)
    # models[0].encoder = torch.load('hash_workspace_obj12_7_8_10/GT_enc.pth')
    # models[0].backbone = torch.load("hash_workspace_obj12_7_8_10/GT_mlp.pth")
    # models[0].encoder=torch.load(f'GT_enc_{n}_{d}_{hashmap_size}.pth')
    # models[0].backbone=torch.load(f'GT_mlp_{n}_{d}_{hashmap_size}.pth')
    
    dim = models[0].encoder.hash_table.shape[0]
    dim2 = dim
    # w = nn.Parameter(torch.randn(dim, dim, device='cuda', requires_grad=True))  #w-commment
    # N = nn.Parameter(torch.randn(dim,dim2, device='cuda', requires_grad=True))
    # M = nn.Parameter(torch.randn(dim2, dim, device='cuda', requires_grad=True))

    # w = torch.load('w2.pth')  
    
    lr = args.lr

    from sdf.provider import SDFDataset
    from loss import mape_loss
    train_datasets = []
    train_loaders = []
    valid_datasets = []
    valid_loaders = []
    # for i in range(_n):
    for path in obj_list:
        train_dataset = SDFDataset(path, size=args.steps_per_epoch, num_samples=args.num_samples,sample_ratio_n=n,sample_ratio_d=d)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_datasets.append(train_dataset)
        train_loaders.append(train_loader)

    criterion = mape_loss # torch.nn.L1Loss()

    enc_optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': model.encoder.parameters()}
        #{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    net_optimizer = lambda model: torch.optim.Adam([
        #{'name': 'encoding', 'params': model.encoder.parameters()},
        {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6}
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    # w_optimizer = lambda w: torch.optim.Adam([
    #     {'name': 'encoding', 'params': [w]}
    #     #{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},       #w-commment
    # ], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    
    # scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    subset_n = _n
    trainer = Trainer('ngp', _n, models,workspace=workspace1, enc_optimizer=enc_optimizer,  net_optimizer=net_optimizer, criterion=criterion, ema_decay=0.95, fp16=False, lr_scheduler=None,scheduler_update_every_step=False, train_net=not args.freeze_decoder)
    trainer.train(train_loaders,args.epochs,subset_n)
    # w_state_dict = w.state_dict()
    # path = f"saved_models/w.pth"
    # torch.save(w_state_dict, path)
    

    # for idx,model in enumerate(models):
    #     # w_state_dict = w.state_dict()
    #     # path = f"saved_models/w.pth"
    #     # torch.save(w_state_dict, path)
    #     torch.save(model.encoder,'shifted_enc.pth')
    # torch.save(models[0].encoder,'GT_enc.pth')
    # torch.save(models[0].backbone,'GT_mlp.pth')
    # torch.save(w,f'w_{n}_{d}.pth')   #w-commnet
    # torch.save(N,workspace1+f'N_{n}_{d}_{hashmap_size}.pth')
    # torch.save(M,workspace1+f'M_{n}_{d}_{hashmap_size}.pth')
    # print("Completed saving path of models")
    
    
    #trainer.save_mesh(os.path.join(workspace1, 'results', 'output.ply'), 1024)
    # save_mesh(models,save_path=workspace1)
        
