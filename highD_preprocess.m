%% Process dataset into mat files %%

clear;
clc;

%% Inputs:
% Locations of raw input files:
highD = './new_data/highd-dataset-v1.0/data/';

%% Fields: 

%{ 
1: Dataset ID
2: Vehicle ID
3: Frame ID
4: Local X
5: Local Y
6: Vehicle Velocity
7: Vehicle Acceleration
8: Lane ID
9: Class
10: Lateral Maneuver
11: Longitudinal Maneuver
12-50: Neighbor Car Ids at grid location
%}

traj = {};
vehTrajs = {};
vehTimes = {};
cnt = 1;

% 4~14 25~57
dataList = [4:14, 25:57];
for i = dataList
    datafile = sprintf('%s%02d_tracks.csv', highD, i);
    traj{cnt} = readmatrix(datafile);

    % 删除负速度
    negXIDs = unique(traj{cnt}(traj{cnt}(:, 7) < 0, 2));
    traj{cnt}(ismember(traj{cnt}(:, 2), negXIDs), :) = [];

    % 筛选前9列
    traj{cnt} = single([cnt * ones(size(traj{cnt},1),1), traj{cnt}]);
    speed = sqrt(traj{cnt}(:, 8).^2 + traj{cnt}(:, 9).^2);
    acceleration = sqrt(traj{cnt}(:, 10).^2 + traj{cnt}(:, 11).^2);
    traj{cnt} = [traj{cnt}(:, [1, 3, 2, 4, 5]), speed, acceleration, traj{cnt}(:, 26)];
    traj{cnt} = [traj{cnt}, 2 * ones(size(traj{cnt}, 1), 1)];

    vehTimes{cnt} = containers.Map;
    vehTrajs{cnt} = containers.Map;

    cnt = cnt + 1;
end 
cnt = cnt - 1;



%% Parse fields (listed above):
disp('Parsing fields...')

for ii = 1:cnt
    % 获取所有车的id
    vehIds = unique(traj{ii}(:,2));

    % 遍历所有车id
    for v = 1:length(vehIds) 
        % 获取某个id的路径数据放vehTrajs(按vehId(v)获取对应的所有行值)
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end

    % 获取所有有车出现的时间戳
    timeFrames = unique(traj{ii}(:,3));

    % 遍历时间戳，将每个时间戳的所有行放vehTimes
    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end

    % 遍历每一行
    for k = 1:length(traj{ii}(:,1))        
        % 获取到一行数据的时间戳, 数据集id, 车辆编号
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId)); % 那么可以通过车辆编号找到这辆车的轨迹数据
        ind = find(vehtraj(:,3)==time); % 因此可以找到这辆车在当前时间戳的下标
        ind = ind(1); 
        lane = traj{ii}(k,8);   % 获取当前行的车道
        
        
       %% Get lateral maneuver:
       % 横向操作分析4s后和4s前的车道和当前车道打标签在第10位，3：左转，2：右转，1：车道不变
        ub = min(size(vehtraj,1),ind+40);
        lb = max(1, ind-40);
        if vehtraj(ub,8)>vehtraj(ind,8) || vehtraj(ind,8)>vehtraj(lb,8)
            traj{ii}(k,10) = 3;
        elseif vehtraj(ub,8)<vehtraj(ind,8) || vehtraj(ind,8)<vehtraj(lb,8)
            traj{ii}(k,10) = 2;
        else
            traj{ii}(k,10) = 1;
        end
        
        
       %% Get longitudinal maneuver:
       % 分析速度-3s~+5s判断是否在刹车
       % upperbound and lowerbound
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,11) =1;  % 速度不变=1
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb); % 历史速度 v_history
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);  % 未来速度 v_future
            if vFut/vHist <0.8      % 历史速度较大
                traj{ii}(k,11) =2;  % 表示在减速=2
            elseif vFut/vHist > 1.25    % 未来速度较大
                traj{ii}(k,11) = 3;     % 表示在加速=3
            else
                traj{ii}(k,11) =1;      % 速度不变=1
            end
        end
        % Get 
        % Get grid locations:
        % 获取周围车辆数据
        t = vehTimes{ii}(int2str(time));    % 获取当前时间戳的所有车辆
        frameEgo = t(t(:,8) == lane,:);     % frameEgo: 当前车道的所有车辆
        frameL = t(t(:,8) == lane-1,:);     % frameL：左侧1车道的所有车辆
        frameR = t(t(:,8) == lane+1,:);     % frameR：右侧1车道的所有车辆
        if ~isempty(frameL)                 % 左侧有车
            for l = 1:size(frameL,1)        % 遍历所有左侧车
                y = frameL(l,5)-traj{ii}(k,5);  % 获取左侧车的local_Y，减去当前车的所有local_Y获得前后距离
                if abs(y) <90               % 90inches内纳入考虑范围
                    gridInd = 1+round((y+90)/15);   % 划分栅格总计180/15=12个，索引位置1~13
                    traj{ii}(k,11+gridInd) = frameL(l,2);   % 从12位开始，列数12~24
                end
            end
        end
        for l = 1:size(frameEgo,1)                      % 当前车道栅格化
            y = frameEgo(l,5)-traj{ii}(k,5);            
            if abs(y) <90 && y~=0                       % 排除自己，90inches
                gridInd = 14+round((y+90)/15);          % 从索引14开始，分12栅格(14~26)
                traj{ii}(k,11+gridInd) = frameEgo(l,2); % 25~37
            end
        end
        if ~isempty(frameR)                 % 右侧车辆数据
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);      % 27~39
                    traj{ii}(k,11+gridInd) = frameR(l,2); % 38~50
                end
            end
        end
        
    end
end

% save('./new_data/allData_s','traj');       % 数据存allData_s.mat


%% Split train, validation, test
% load('./new_data/allData_s','traj');
disp('Splitting into train, validation and test sets...')

tracks = {};
trajAll = [];
for k = 1:cnt
    vehIds = unique(traj{k}(:, 2)); % 所有车辆id
    for l = 1:length(vehIds)        
        vehTrack = traj{k}(traj{k}(:, 2)==vehIds(l), :);    % 找当前车辆id的所有行
        tracks{k,vehIds(l)} = vehTrack(:, 3:11)'; % 包含车辆全时间所有信息，以及横向纵向操作信息
        filtered = vehTrack(30+1:end-50, :);    % 去除前30行和最后50行，3s历史数据, 5s预测数据
        
        trajAll = [trajAll; filtered];
    end
end
clear traj;

trajTr=[];
trajVal=[];
trajTs=[];
for ii = 1:cnt
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
save('./new_data/TrainSet','traj','tracks');

traj = trajVal;
save('./new_data/ValSet','traj','tracks');

traj = trajTs;
save('./new_data/TestSet','traj','tracks');
