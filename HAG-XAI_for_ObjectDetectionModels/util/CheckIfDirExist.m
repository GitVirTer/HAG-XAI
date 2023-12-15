function CheckIfDirExist(Path)

PathField = fieldnames(Path);
for iPath = 1:numel(PathField)
    curPath = getfield(Path, PathField{iPath});
    if ~exist(curPath,'dir')
        mkdir(curPath);
        disp(['Created ' curPath]);
    end
    
end

end