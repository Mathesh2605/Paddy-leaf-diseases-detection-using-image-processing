classdef FeatureExt
    methods (Static)
        function [X_data]= FeatExtract(ds)
            imcnt=1;
            step=0;
            z=0;
            while hasdata(ds) 
                a = read(ds) ;
                gr=rgb2gray(a);
%                 norm=mat2gray(gr);
                norm=double(gr);
               % disp(norm)
                figure('Name','Input'),imshow(imresize(a,0.3)),title('Input Image')
                resize=imresize(norm,[128 128]);
%                 figure('Name','Resizing'),imshow(resize),title('Resizing')
        
                % Select region interactively
                mask=zeros(size(resize));
                mask(25:end-25,25:end-25) = 1;
                %     mask = roipoly;
    
                figure('Name','Mask'), imshow(mask)
                title('Initial MASK');
  
                % Segment the image using active contours
                maxIterations = 300; % More iterations may be needed to get accurate segmentation. 
                bw = activecontour(resize, mask, maxIterations, 'Chan-Vese');
  
                % Display segmented image
                figure('Name','Segmented'), imshow(bw)
                title('Segmented Image');
            
                fcnt=1;
                values=imresize(resize,[1,50]);
                feat(imcnt,:)=values;
%                 for i = 1:128
%                     for j = 1:128
%                         values=resize(i,j);
%                         feat(imcnt,fcnt)=values;
%                         fcnt=fcnt+1;
%                     end
%                 end

                X_data=feat;
                Y_data(1, imcnt)=double(step);
                if(z==4)
                    step=step+1;
                    z=0;
                end
                imcnt=imcnt+1;
            end

            feat=abs(feat);
            X_data=feat(1,1:50);
            %         disp(feat)
            %         save('Features.mat','feat')
            %         disp('Feature Extracted for All Images')
        end
    end
end