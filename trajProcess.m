load('./new_data/allData_s','traj'); 
disp('Splitting into train, validation and test sets...')

tracks = {};
trajAll = [];
for k = 1:6
    vehIds = unique(traj{k}(:, 2)); % 所有车辆id
    for l = 1:length(vehIds)        
        vehTrack = traj{k}(traj{k}(:, 2)==vehIds(l), :);    % 找当前车辆id的所有行
        tracks{k,vehIds(l)} = vehTrack(:, 3:11)'; % 包含车辆全时间所有信息，以及横向纵向操作信息

        
        % 同道判断
        if vehTrack(40+1, 8) ~= vehTrack(end-50, 8)
            filtered = vehTrack(30+1:end-50, :);    % 去除前30行和最后50行，3s历史数据, 5s预测数据
            trajAll = [trajAll; filtered];
        end
        
    end
end
clear traj;

trajTr=[];
trajVal=[];
trajTs=[];
for ii = 1:6
    no = trajAll(find(trajAll(:,1)==ii),:);
    len1 = length(no)*0.7;
    len2 = length(no)*0.8;
    trajTr = [trajTr;no(1:len1,:)];
    trajVal = [trajVal;no(len1:len2,:)];
    trajTs = [trajTs;no(len2:end,:)];
end

disp('Saving mat files...')
%%
traj = trajTr;
save('./new_data/TrainSet_ngsim','traj','tracks');

traj = trajVal;
save('./new_data/ValSet_ngsim','traj','tracks');

traj = trajTs;
save('./new_data/TestSet_ngsim','traj','tracks');
