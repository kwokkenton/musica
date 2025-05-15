import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from muse.data import DataProcessor, MaestroDataset, MaestroDatasetSingle
from muse.model import calculate_accuracy, get_decoder_inputs_and_targets
from muse.utils import count_trainable_params, get_device, get_wandb_checkpoint_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


class Validator:
    def __init__(self, validation_dataloader: DataLoader, device: torch.device):
        self.valid_dl = validation_dataloader
        self.device = device

    def validate(self, model: torch.nn.Module, loss_fn: torch.nn.Module):
        total_correct = 0
        count = 0
        total_loss = 0.

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_dl)):
                x, y = batch

                # THIS IS EXACTLY THE SAME CODE AS THE FORWARD -----------------

                inputs, targets = get_decoder_inputs_and_targets(
                    y,
                    model.eos_id,
                    model.pad_id,
                )
            
                x = x.to(self.device)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Scores are (unnormalised) logits
                scores = model(x, inputs)

                # Then do the loss
                loss = loss_fn(
                    scores.view(targets.numel(), -1),
                    targets.view(-1),
                )
                # -------------------------------------------------------------------

                total_loss += loss.item()

                # SAME CODE AS IN PER BATCH EVALUATION  -----------------
                _, correct = calculate_accuracy(
                    scores, targets, pad_token_id=model.pad_id,
                )
                # -------------------------------------------------------------------
                total_correct += correct
                count += torch.numel(targets)

        return total_loss / len(self.valid_dl), total_correct / count


class Trainer:
    def __init__(
        self,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        setup_config: dict,
        device: torch.device,
    ):

        self.setup_config = setup_config
        self.batch_size = setup_config.get('batch_size')
        self.num_workers = setup_config.get('num_workers')

        self.device = device

        # Set up datasets and dataloaders

        self.train_dl = train_dl
        self.val_dl = val_dl
        # self.val_dl = DataLoader(
        #     self.val_ds,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     drop_last=True,
        #     num_workers=self.num_workers,
        #     collate_fn=collate_fn,
        # )

        self.validator = Validator(self.val_dl, self.device)

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 100,
    ):

        running_loss = 0.
        last_loss = 0.

        model = model.to(self.device)

        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            # Zero your gradients for every batch!
            optimiser.zero_grad()
            # y is a padded sequence
            # x shape B, D
            x, y = batch

            # CHANGE THIS-------------------------------------------------------
            # Training forward code
            inputs, targets = get_decoder_inputs_and_targets(
                    y,
                    model.eos_id,
                    model.pad_id,
                )
            
            x = x.to(self.device)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # Scores are (unnormalised) logits
            scores = model(x, inputs)

            # Then do the loss
            loss = loss_fn(
                scores.view(targets.numel(), -1),
                targets.view(-1),
            )
            # -------------------------------------------------------------------
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):

                # CHANGE THIS---------------------------------------------------
                # Logs and sanity checking code
                logger.info(
                    f'Correct seq:\t{targets[0]}',
                )
                logger.info(
                    f'Predicted seq:\t{scores.argmax(-1)[0]}',
                )

                # with torch.no_grad():
                #     # Predict first token
                #     generated = model.forward_sequential(x[0:1])
                #     logger.info(
                #         f'Sequentially predicted seq:\t{model.text_tokenizer.decode(generated[0])}',
                #     )

                # Calculate accuracy metric
                accuracy, _ = calculate_accuracy(
                    scores, targets, pad_token_id=model.pad_id,
                )
                ppl = torch.exp(loss)
                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  For batch {batch_idx + 1}, the loss is {last_loss}, the accuracy is {accuracy}, the perplexity is {ppl}, ',
                )
                # -------------------------------------------------------------------
                running_loss = 0.

                # In case you want to save every printed time
                #    checkpoint = {
                #     'model_state_dict': model.state_dict(),
                #     'optimiser_state_dict': optimiser.state_dict(),
                # }
                # checkpoint_path = os.path.join(
                #         '/Users/kenton/projects/mlx-institute/transformer/checkpoints',
                #         f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                #     )
                #     # torch.save(checkpoint, checkpoint_path)

        return last_loss, accuracy

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        config: dict,
    ):

        logger.info(f'Training config: {config}')
        log_to_wandb = config.get('log_to_wandb')
        log_locally = config.get('log_locally')
        checkpoint_folder = config.get(
            'checkpoint_folder',
        )
        project_name = config.get('project_name')
        model_name = config.get('model_name')
        epochs_per_log = 10

        if log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project=project_name,
                config=config,
            )

        try: 
            for epoch in range(epochs):
                logger.info(f'Training: Epoch {epoch + 1} of {epochs}')
                train_loss, train_accuracy = self.train_one_epoch(
                    model, loss_fn, optimiser, config.get(
                        'batches_print_frequency',
                    ),
                )
                logger.info(
                    f'Validating: Epoch {epoch + 1} of {epochs}.',
                )
                # Run validation to sanity check the model
                val_loss, val_accuracy = self.validator.validate(
                    model, loss_fn,
                )
                logger.info(
                    f'Epoch {epoch + 1} of {epochs} train loss: {train_loss}'
                    f'train accuracy: {train_accuracy} val loss: {val_loss}'
                    f'val accuracy: {val_accuracy}',
                )

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                        }
                
                if log_to_wandb:
                    wandb.log({
                        'train/loss': train_loss,
                        'train/accuracy': train_accuracy,
                        'val/loss': val_loss,
                        'val/accuracy': val_accuracy,
                    })

                if epoch%epochs_per_log == (epochs_per_log - 1):
                    if log_locally or log_to_wandb:
                        if log_locally:
                            checkpoint_path = os.path.join(
                                checkpoint_folder,
                                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                            )
                            torch.save(checkpoint, checkpoint_path)

            

                    if log_to_wandb:
                        checkpoint_path = os.path.join(
                            wandb.run.dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                        )
                        torch.save(checkpoint, checkpoint_path)
                        artifact = wandb.Artifact(model_name, type='checkpoint')
                        artifact.add_file(checkpoint_path)
                        wandb.run.log_artifact(artifact)

        # Save in case it crashes
        except Exception as e:  
            logger.error(f'Crashed {e}')
            if log_locally:
                checkpoint_path = os.path.join(
                    checkpoint_folder,
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                )
                torch.save(checkpoint, checkpoint_path)

            if log_to_wandb:
                checkpoint_path = os.path.join(
                    wandb.run.dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                )
                torch.save(checkpoint, checkpoint_path)
                artifact = wandb.Artifact(model_name, type='checkpoint')
                artifact.add_file(checkpoint_path)
                wandb.run.log_artifact(artifact)
        finally:
            wandb.finish()


if __name__ == '__main__':

    import argparse

    from muse.model import MusicTranscriber

    def parse_args():
        parser = argparse.ArgumentParser(
            description='Example script with a --log_to_wandb flag',
        )
        parser.add_argument(
            '--epochs',
            default=10,
            type=int
        )
        parser.add_argument(
            '--batch_size',
            default=2,
            type=int
        )

        parser.add_argument(
            '--log_to_wandb',
            action='store_true',
            help='If set, enable logging to Weights & Biases',
        )
        parser.add_argument(
            '--log_locally',
            action='store_true',
            help='If set, enable logging locally.',
        )
        parser.add_argument(
            '--wandb_checkpoint',
            type=str,
            required=False,
            help="Wandb identifier, e.g., 'kwokkenton-individual/mlx-week2-search-engine/towers_rnn:latest'",
        )

        parser.add_argument(
            '--path_to_data_dir',
            type=str,
            required=True,
            help='If set, enable logging to Weights & Biases',
        )

        parser.add_argument(
            '--path_to_csv',
            type=str,
            required=True,
            help='If set, enable logging to Weights & Biases',
        )

        return parser.parse_args()

    args = parse_args()
    log_to_wandb = args.log_to_wandb
    wandb_checkpoint = args.wandb_checkpoint
    path_to_data_dir = args.path_to_data_dir
    path_to_csv = args.path_to_csv
    epochs = args.epochs
    batch_size = args.batch_size
    log_locally = args.log_locally


    # midi_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    # wav_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.wav'

    dp = DataProcessor(batch_size=batch_size)

    train_ds = MaestroDataset(path_to_data_dir, path_to_csv, dp)
    val_ds = MaestroDataset(path_to_data_dir, path_to_csv, dp, train=False)

    model_config = {'d_model': 256,
                'enc_dims': dp.ap.n_mels, 
                'enc_max_len': dp.max_enc_len,
                'dec_max_len': dp.max_dec_len,
                'dec_vocab_size': dp.tok.vocab_size,
                'eos_id':dp.tok.eos_id,
                'bos_id':dp.tok.bos_id,
                'pad_id':dp.tok.pad_id}
    
    # Config parameters
    setup_config = {
        'batch_size': batch_size,
        'num_workers': 1,
    }

    # Training configs
    training_config = {
        'project_name': 'mlx-week5-music',
        'model_name': 'transcriber-256',
        'epochs': epochs,
        'lr': 5e-5,
        'log_locally': log_locally,
        'log_to_wandb': log_to_wandb,
        'batches_print_frequency': 1,
        'checkpoint_folder': 'checkpoints',
    }

    device = get_device()

    model = MusicTranscriber(model_config)

    # One song per batch
    train_dl = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=setup_config.get('num_workers'),
            collate_fn=dp.collate_fn,
        )
    
    val_dl = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=setup_config.get('num_workers'),
            collate_fn=dp.collate_fn,
        )

    num_params = count_trainable_params(model)
    logger.info(f'There are {num_params} trainable parameters in the model.')
    logger.info(model)


    optimiser = torch.optim.Adam(
        model.trainable_params(), lr=training_config.get('lr'),
    )

    # Load previously trained model
    checkpoint_path = '/root/musica/checkpoints/20250515_134856.pth'
    # if wandb_checkpoint:
    #     checkpoint_path = get_wandb_checkpoint_path(
    #         wandb_checkpoint,
    #     )

    # Load the model
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ignore pad id
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.pad_id)
    trainer = Trainer(
        train_dl=train_dl,
        val_dl=val_dl,
        setup_config=setup_config,
        device=device,
    )
    # trainer.train_one_epoch(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimiser=optimiser,
    #     batches_print_frequency=training_config.get('batches_print_frequency'),
    # )

    trainer.train(
        epochs=training_config.get('epochs'),
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        config=training_config,
    )
