import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import wandb
from tqdm import tqdm

from utils.blend import blend_cpu

from external.voxel2mesh.utils.utils_common import DataModes, mkdir, append_line, write_lines
from external.voxel2mesh.utils.utils_voxel2mesh.file_handle import save_to_obj
from external.voxel2mesh.utils.rasterize.rasterize import Rasterize


class Structure(object):

    def __init__(self, voxel=None, mesh=None, points=None, sphr_mesh=None, malignant=None):
        self.voxel = voxel
        self.mesh = mesh
        self.sphr_mesh = sphr_mesh
        self.points = points
        self.malignant = malignant


class Evaluator(object):
    def __init__(self, net, optimizer, data, save_path, config, support):
        self.data = data
        self.net = net
        self.current_best = None
        self.save_path = save_path + '/best_performance'
        self.latest = save_path + '/latest'
        self.optimizer = optimizer
        self.config = config
        self.support = support
        self.count = 0

    def write_to_wandb(self, epoch, split, performences, num_classes):
        if self.config.wab:
            log_vals = {"epoch": epoch}
            for key, value in performences[split].items():
                log_vals[split + '_' + key + '/mean'] = np.nanmean(performences[split][key])
                try:
                    for i in range(1, num_classes):
                        log_vals[split + '_' + key + '/class_' + str(i)] = np.nanmean(performences[split][key][:, i - 1])
                except: # measures without classes
                    pass
            try:
                wandb.log(log_vals)
            except Exception as ex:
                print(ex)

    def save_model(self, epoch):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path + '/model.pth')

    def evaluate(self, epoch, writer=None, backup_writer=None):
        self.net = self.net.eval()
        performences = {}
        predictions = {}

        for split in [DataModes.VALIDATION]:
            dataloader = DataLoader(
                self.data[split], batch_size=1, shuffle=False)
            performences[split], predictions[split] = self.evaluate_set(
                dataloader)

            self.write_to_wandb(epoch, split, performences, self.config.num_classes)

        if epoch > 10 and self.support.update_checkpoint(best_so_far=self.current_best, new_value=performences):
            mkdir(self.save_path)
            mkdir(self.save_path + '/mesh')
            mkdir(self.save_path + '/voxels')

            self.save_model(epoch)
            self.save_results(predictions[DataModes.VALIDATION], epoch,
                              performences[DataModes.VALIDATION], self.save_path, split)
            self.current_best = performences

    def evaluate_all(self, epoch, writer=None, backup_writer=None):
        self.net = self.net.eval()
        performences = {}
        predictions = {}

        for split in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            try:
                dataloader = DataLoader(
                    self.data[split], batch_size=1, shuffle=False)
                performences[split], predictions[split] = self.evaluate_set(
                    dataloader)

                self.write_to_wandb(
                    epoch, split, performences, self.config.num_classes)
                if split == DataModes.TESTING:
                    self.data[split].metadata.Malignancy = self.data[split].metadata.Malignancy_weak
                    print("\n LIDC 72 RM")
                    dataloader = DataLoader(
                        self.data[split], batch_size=1, shuffle=False)
                    performences[split], predictions[split] = self.evaluate_set(
                        dataloader)
                    self.write_to_wandb(
                        epoch, split, performences, self.config.num_classes)
            except:
                continue

            mkdir(self.save_path)
            mkdir(self.save_path + '/mesh')
            mkdir(self.save_path + '/voxels')

            self.save_model(epoch)
            self.save_results(
                predictions[split], epoch, performences[split], self.save_path, split)

    def predict(self, data, config):
        name = config.name
        if name == 'unet':
            with torch.no_grad():
                y_hat = self.net(data)
                y_hat = torch.argmax(y_hat, dim=1)

                x = data['x']
                y = Structure(voxel=data['y_voxels'])
                y_hat = Structure(voxel=y_hat)

        elif name == 'voxel2mesh':

            x = data['x']
            with torch.no_grad():
                pred, output = self.net(data)

                pred_meshes = []
                sphr_meshes = []
                true_meshes = []
                true_points = []
                pred_voxels = torch.zeros_like(x)[:, 0].detach().long()
                # embed()
                for c in range(self.config.num_classes-1):
                    # embed()

                    pred_vertices = pred[c][-1][0].detach().data
                    pred_faces = pred[c][-1][1].detach().data
                    sphr_vertices = pred[c][-1][4].detach().data
                    true_vertices = data['vertices_mc'][c].data
                    true_faces = data['faces_mc'][c].data

                    pred_meshes += [{'vertices': pred_vertices,
                                     'faces': pred_faces, 'normals': None}]
                    sphr_meshes += [{'vertices': sphr_vertices,
                                     'faces': pred_faces, 'normals': None}]
                    true_meshes += [{'vertices': true_vertices,
                                     'faces': true_faces, 'normals': None}]
                    true_points += [data['surface_points'][c].data]

                    _, _, D, H, W = x.shape
                    shape = torch.tensor([D, H, W]).int().cuda(config.device)
                    rasterizer = Rasterize(shape)
                    pred_voxels_rasterized = rasterizer(
                        pred_vertices, pred_faces).long()

                    pred_voxels += pred_voxels_rasterized

                true_voxels = data['y_voxels'].data

                x = x.detach().data
                target = data['metadata']['Malignancy']
                y = Structure(mesh=true_meshes, voxel=true_voxels,
                              points=true_points, malignant=target[0])
                y_hat = Structure(mesh=pred_meshes, voxel=pred_voxels,
                                  sphr_mesh=sphr_meshes, malignant=output[1])

        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return x, y, y_hat

    def evaluate_set(self, dataloader):
        performance = {}
        predictions = []

        labels = []
        preds = []
        with tqdm(dataloader, unit="sample") as teval:
            for i, data in enumerate(teval):
                teval.set_description(f"Evaluation")
                x, y, y_hat = self.predict(data, self.config)
                result = self.support.evaluate(y, y_hat, self.config)

                labels.append(y.malignant.detach().cpu().numpy())
                preds.append(y_hat.malignant.detach().cpu().numpy())

                predictions.append((x, y, y_hat))

                for key, value in result.items():
                    if key not in performance:
                        performance[key] = []
                    performance[key].append(result[key])
                    
                #teval.set_postfix(**performance)

        labels = np.asarray(labels)
        preds = np.asarray(preds)
        fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
        #t = thresholds[np.sqrt((1-fpr)**2+tpr**2).argmax()]
        t = 0.5
        auc_ = auc(fpr, tpr)
        CM = confusion_matrix(labels, preds > t, labels=[0, 1])
        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM)/np.sum(CM))
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        precision = tp/(tp+fp)

        performance['auc'] = auc_
        performance['accuracy'] = acc
        performance['sensitivity'] = sensitivity if sensitivity is not None else 0
        performance['specificity'] = specificity if specificity is not None else 0

        print('\nTestset Accuracy(mean): %f%%' % (100 * acc), end=", ")
        print('Testset AUC: %f' % auc_)
        print()
        print('Confusion Matirx : ')
        print(CM)
        print('Sensitivity : ', (tp/(tp+fn))*100, end=", ")
        print('Specificity : ', (tn/(tn+fp))*100, end=", ")
        print('Precision: ', (tp/(tp+fp))*100, end=", ")
        print('NPV: ', (tn/(tn+fn))*100, end=", ")
        print('F1 : ', ((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()

        for key, value in performance.items():
            performance[key] = np.array(performance[key])
        return performance, predictions

    def save_results(self, predictions, epoch, performence, save_path, split):
        xs = []
        ys_voxels = []
        ys_points = []
        y_hats_voxels = []
        y_hats_points = []
        y_hats_meshes = []

        mode = f'/{split}_'

        for i, data in enumerate(predictions):
            x, y, y_hat = data
            pid = self.data[split][i]['pid']

            xs.append(x[0, 0])

            if y_hat.points is not None:
                for p, (true_points, pred_points) in enumerate(zip(y.points, y_hat.points)):
                    save_to_obj(save_path + '/points/' + mode + 'true_' +
                                pid + '_part_' + str(p) + '.obj', true_points, [])
                    if pred_points.shape[1] > 0:
                        save_to_obj(save_path + '/points/' + mode + 'pred_' +
                                    pid + '_part_' + str(p) + '.obj', pred_points, [])

            if y_hat.mesh is not None:
                for p, (true_mesh, pred_mesh, sphr_mesh) in enumerate(zip(y.mesh, y_hat.mesh, y_hat.sphr_mesh)):
                    save_to_obj(save_path + '/mesh/' + mode + 'true_' + pid + '_part_' + str(
                        p) + '.obj', true_mesh['vertices'], true_mesh['faces'], true_mesh['normals'])
                    save_to_obj(save_path + '/mesh/' + mode + 'sphr_' + pid + '_part_' + str(
                        p) + '.obj', sphr_mesh['vertices'], sphr_mesh['faces'], sphr_mesh['normals'])
                    save_to_obj(save_path + '/mesh/' + mode + 'pred_' + pid + '_part_' + str(
                        p) + '.obj', pred_mesh['vertices'], pred_mesh['faces'], pred_mesh['normals'])

            if y_hat.voxel is not None:
                ys_voxels.append(y.voxel[0])
                y_hats_voxels.append(y_hat.voxel[0])

        if performence is not None:
            for key, value in performence.items():
                try:
                    performence_mean = np.mean(performence[key], axis=0)
                    summary = ('{}: ' + ', '.join(['{:.8f}' for _ in range(
                        self.config.num_classes-1)])).format(epoch, *performence_mean)
                    append_line(save_path + mode + 'summary' +
                                key + '.txt', summary)
                    print(('{} {}: ' + ', '.join(['{:.8f}' for _ in range(
                        self.config.num_classes-1)])).format(epoch, key, *performence_mean))

                    all_results = [('{}: ' + ', '.join(['{:.8f}' for _ in range(self.config.num_classes-1)])).format(
                        *((i+1,) + tuple(vals))) for i, vals in enumerate(performence[key])]
                    write_lines(save_path + mode + 'all_results_' + key + '.txt', all_results)
                except: # measures without classes
                    pass

        xs = torch.cat(xs, dim=0).cpu()
        if y_hat.voxel is not None:
            ys_voxels = torch.cat(ys_voxels, dim=0).cpu()
            y_hats_voxels = torch.cat(y_hats_voxels, dim=0).cpu()

            y_hats_voxels = F.upsample(y_hats_voxels[None, None].float(), size=xs.shape)[
                0, 0].long()
            ys_voxels = F.upsample(ys_voxels[None, None].float(), size=xs.shape)[
                0, 0].long()

            y_overlap = y_hats_voxels.clone()
            y_overlap[ys_voxels == 1] = 5
            y_overlap[ys_voxels == 2] = 6
            y_overlap[(ys_voxels == 1) & (y_hats_voxels == 1)] = 3
            y_overlap[(ys_voxels == 2) & (y_hats_voxels == 2)] = 4

            overlay_y_hat = blend_cpu(
                xs, y_hats_voxels, self.config.num_classes)
            overlay_y = blend_cpu(xs, ys_voxels, self.config.num_classes)
            overlay_overlap = blend_cpu(xs, y_overlap, 6)
            overlay = np.concatenate(
                [overlay_y, overlay_y_hat, overlay_overlap], axis=2)
            io.imsave(save_path + mode + 'overlay_y_hat.tif', overlay)
