#BASE VAE IN MODELS.PY! 


def kl_loss(self, mu, logstd, z_sample):
    # Manual formula for kl divergence (more numerically stable!)
    kl_div = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1.0 - 2 * logstd)
    loss = kl_div.sum() / z_sample.shape[0]
    return loss

# VAE forward: 


kl_loss = self.kl_loss(mu, logstd, z_sample)

reconstruction_loss = self.decoder_loss(z_sample, x)

 elif self.metric_loss == 'triplet':
                triplet_loss = TripletLossTorch(
                    threshold=self.metric_loss_kw['threshold'],  ## default = 0.1 !! 
                    margin=self.metric_loss_kw.get('margin'), ## default = 1? 
                    soft=self.metric_loss_kw.get('soft'), # default = true 
                    eta=self.metric_loss_kw.get('eta') # default = 'eta':0.05 # BUT IN PAPER THEY USE ETA = 0!!! 
                    # make sure you're using eta = 0
                )
                metric_loss = triplet_loss(z_sample, y)


# beta_metric_loss is a hyperparam 

loss = reconstruction_loss + beta * kl_loss + self.beta_metric_loss * metric_loss
return loss