function sae = saesetup(size,opts)
    for u = 2 : numel(size)
      if exist('opts','var')
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)],opts);
      else
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);
      end
    end
end
