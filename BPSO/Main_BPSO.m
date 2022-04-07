

clc;
clear;
close all;

%% Load Data


data = readtable('cleaned_data.csv');
Inputs = data(:,1:end-1);
Targets = data(:,end);
Inputs = cat(2, table2array(Inputs(:,[1, 3:end-1])),...
    double(categorical(table2array(Inputs(:,2)))),...
    double(categorical(table2array(Inputs(:,end)))));

Nans = sum(isnan(Inputs),2);
Inputs(Nans==1,:) = [];
X = Inputs;

%% Normalize

MIN = min(X);
MAX = max(X);
X = (X-MIN)./(MAX-MIN);

Nans = sum(isnan(X),1);
disp(['Removed features: ', num2str(find(Nans))])
X(:,find(Nans)) = []; %#ok

%% Problem Definition

ClusteringMetric = 'CS'; % 'DB' or 'CS'
CostFunction=@(s) ClusteringCost(s, X, ClusteringMetric);     % Cost Function
VarSize=[1 size(X,2)];  % Decision Variables Matrix Size
nVar=prod(VarSize);     % Number of Decision Variables
VarMin= zeros(VarSize);      % Lower Bound of Variables
VarMax= ones(VarSize);      % Upper Bound of Variables

%% PSO Parameters

MaxIt=150;      % Maximum Number of Iterations  
nPop=200;       % Population Size (Swarm Size)  
w=1;            % Inertia Weight  
wdamp=0.99;     % Inertia Weight Damping Ratio 
c1=2;           % Personal Learning Coefficient  
c2=1.5;         % Global Learning Coefficient   


% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Out=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Out=[];

particle=repmat(empty_particle,nPop,1);

BestSol.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost, particle(i).Out]=CostFunction(round(particle(i).Position));
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Out=particle(i).Out;
    
    % Update Global Best
    if particle(i).Best.Cost<BestSol.Cost
        
        BestSol=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);


%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(BestSol.Position-particle(i).Position);
        
        % Logistic sigmoid transformation function (S)
        particle(i).Velocity = 1./(1+exp(particle(i).Velocity));
     
        % Update Position
        particle(i).Position = double(unifrnd(0,1,VarSize)<particle(i).Velocity);

        
        % Evaluation
        [particle(i).Cost, particle(i).Out] = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Out=particle(i).Out;
            
            % Update Global Best
            if particle(i).Best.Cost<BestSol.Cost
                
                BestSol=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=BestSol.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

%% Results

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

%% Show results

Features = find(BestSol.Out.Select==1);
disp(['Features which are selected = ' num2str(Features)]);

%% Save results

Selectedfeature = Features';
Table = table(Selectedfeature);
writetable(Table, 'Result.xlsx', 'sheet', 'Selected features')
ClusteringResult = BestSol.Out.ind;
Table = table(ClusteringResult);
writetable(Table, 'Result.xlsx', 'sheet', 'Clustering result')


