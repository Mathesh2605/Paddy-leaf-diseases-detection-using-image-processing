clc
clear all
close all
location = '.\Dataset\*.jpg';
ds = imageDatastore(location)  ;%  Creates a datastore for all images in your folder
imcnt=1;
step=0;
z=0;
while hasdata(ds) 
    z=z+1;
    a = read(ds) ;  
    gr=rgb2gray(a);
    
    norm=double(gr);
   
    resize=imresize(norm,[128 128]);
     % Select region interactively
    mask=zeros(size(resize));
    mask(25:end-25,25:end-25) = 1;
    %     mask = roipoly;
    %   
    %     figure, imshow(mask)
    %     title('Initial MASK');
  
    % Segment the image using active contours
    maxIterations = 300; % More iterations may be needed to get accurate segmentation. 
    bw = activecontour(resize, mask, maxIterations, 'Chan-Vese');
  
    % Display segmented image
    %     figure, imshow(bw)
    %     title('Segmented Image');
    fcnt=1;
    values=imresize(resize,[1,50]);
    feat(imcnt,:)=values;
            
%     for i = 1:128
%         for j = 1:128
%             values=imresize(resize,[1,50]);
%             feat(imcnt,:)=values;
%             fcnt=fcnt+1;
%         end
%     end
    %     feat(imcnt,fcnt)=values;
    %     fcnt=fcnt+1;
    

    X_data=feat;
    Y_data(1, imcnt)=double(step);
    if(z==4)
        step=step+1;
        z=0;
    end
    imcnt=imcnt+1;
end
feat=abs(feat);
disp(feat)
save('Features.mat','feat')
disp('Feature Extracted for All Images')

X=X_data;
save('X_data.mat','X')

Y=Y_data';
save('Y_data.mat','Y')

[coeff,score,~,~,explained,mu] = pca(X);
idx = find(cumsum(explained)>74.265,1);

scoreTrain99 = score(:,1:idx);
% %SVM classifier
%  svm_classifier = fitcecoc(scoreTrain99,Y);
%  disp('SVM Classifier')
%  disp(svm_classifier)
%  
X_data=X_data(:,1:50);
svm_classifier = fitcecoc(X_data,Y);
disp('SVM Classifier')
disp(svm_classifier)
 
label = predict(svm_classifier,X_data);
disp('Predict SVM')
%  label = predict(svm_classifier,scoreTrain99);
%  disp('Predict SVM')
disp(label)
error = resubLoss(svm_classifier);
confusion = confusionmat(Y,label);
disp('Confusion Matrix:')
disp(confusion)
disp('Error:')
disp(error)
EVAL=Evaluate(Y,label);
[FileName, FilePath] = uigetfile('*.*');
if ~ischar(FileName)
   return;
end
location = fullfile(FilePath, FileName);
%  location = '.\Dataset\normal2.jpg';       %  folder in which your images exists
ds = imageDatastore(location)  ;
one_img_data=FeatureExt.FeatExtract(ds);
%  [coeff1,score1,~,~,explained1,mu1] = pca(one_img_data);
%  idx1 = find(cumsum(explained1)>99.999,1);
%  scoreTrain= score1(:,1:12);
%  
%  svm_classifier = fitcecoc(scoreTrain,Y);
%  disp('SVM Classifier')
%  disp(svm_classifier)
 
disp(svm_classifier)
%  disp('Predict SVM')
[q, idx] = ismember(one_img_data, X_data, 'rows')
% label
label = predict(svm_classifier,one_img_data);
disp('Predict SVM')
disp(label)
if(label==0)
    disp('LEAF BLIGHT')
end
if(label==1)
    disp('BROWN SPOT')
end
if(label==2)
    disp('HEALTHY LEAF')
end
if(label==3)
    disp('LEAF BLAST')
end
if(label==4)
    disp('LEAF SMUT')
end
