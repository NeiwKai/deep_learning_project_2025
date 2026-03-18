import torch

def train_one_epoch(
        dataloader, model, losses, optimizer, 
        epoch, device, writer, log_step_interval=50
    ):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0

    for i, (X, y, bboxes) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()

        cl_pred, bb_pred = model(X)

        cl_loss = losses["cl_head"](cl_pred, y)
        bb_loss = losses["bb_head"](bb_pred, bboxes)

        total_loss = cl_loss + bb_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if (i+1) % log_step_interval == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Class: {cl_loss.item():.4f}, BBox: {bb_loss.item():.4f})")
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(dataloader) + i)

def test(dataloader, model, losses, device):
    num_batches = len(dataloader)
    model.eval()
    cl_loss, bb_loss = 0, 0
    y_preds, y_trues = [], []
    bbox_preds, bbox_trues = [], []

    with torch.no_grad():
        for i, (X, y, bboxes) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)
            bboxes = bboxes.to(device)

            cl_pred, bb_pred = model(X)

            cl_loss += losses["cl_head"](cl_pred, y)
            bb_loss += losses["bb_head"](bb_pred, bboxes)

            y_preds.append(cl_pred.argmax(1))
            y_trues.append(y)
            bbox_preds.append(bb_pred)
            bbox_trues.append(bboxes)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    bbox_preds = torch.cat(bbox_preds)
    bbox_trues = torch.cat(bbox_trues)

    cl_loss /= num_batches
    bb_loss /= num_batches
    return cl_loss, bb_loss, y_preds, y_trues, bbox_preds, bbox_trues
