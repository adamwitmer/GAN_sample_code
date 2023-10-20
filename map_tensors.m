close all, clear all, clc;

% disp(sprintf('Processing Exp # %d, Cond: %d, Video: %d, Frame %d', i, j, n, k));

folder = '/Use/z'[9] C3 HD70 10^-5 M'}; %'[7] A3 HD70 CN'}; %,'[8] A3 HD70 10^-4 M',...
%    '[9] C3 HD70 10^-5 M'};
file_info = {};

for i = 1:length(experimental_dirs)

    for j = 5

        vid_numbers = {'1','2','3','4','5'};
        cd(strcat(basefolder, experimental_dirs{i},'/', vid_numbers{j}));
        imgs = dir('I*.jpg');

        for k = 1:length(imgs)

%             close all;

            img = imread(imgs(k).name);
            img = rgb2gray(img);
            img_adjust = imfilter( img, ...
            fspecial( 'gaussian', 7 ), ...
            'symmetric' ); %(1+2*2*filtSize)
            figure; imshow(img);

            temp = entropyfilt( img_adjust, true((3)) );
            temp = temp - min(min( temp ));
            temp = temp ./ max(max( temp ));
            imshow(temp);
            keyboard

            temp = imopen( temp, ...
            strel('disk',7) );
            imshow(temp);
            keyboard;

            temp = temp >= 0.45; % binarize
            imshow(temp);
            keyboard;

            temp = imfill(temp, 'holes');
            imshow(temp);
            keyboard;

%                 temp = bwareaopen(temp, 3000);
%             figure(f2), imshow(temp);

%                 temp = imerode(temp, strel('disk',1));
%             figure(f2), imshow(temp);

            img_final = bwareaopen(temp, 2000);

%                 f1 = figure;
%                 f2 = figure;
%                 figure(f1), imshow(img);
%                 figure(f2), imshow(img_final);
%                 keyboard;
%             end

%                 Uncomment for plotting boundary pixels for image mask
            keyboard;
            boundary.x = bwmorph(img_final, 'remove');
            [row,col] = find(boundary.x);
%             imshow(img); hold on;
%             plot(col, row, '.c');




            %         Uncomment to view boundary outlines on original image
            %         boundary(i).x = bwmorph(img_final, 'remove');
            %         [row,col] = find(boundary(i).x);
            % fig = figure;

            imshow(img); hold on;
            plot(col, row, '.c');
            export_fig(sprintf('boundary %d.jpg',b));
            keyboard;
            % close all;
            % figure; imshow(img);
%             close all;



            % %     double comment for connected component analysis
            cc = bwconncomp(img_final);
            C(k).x = regionprops(cc,'centroid');
            B(k).x = regionprops(cc,'boundingbox');
            A(k).x = regionprops(cc,'Area');
            %
            %index centroid values, plot on original image, save image

            %index centroid values, to be used to find crop bounding box.
            for w = 1:length(C(k).x(:,1))
                c(:,w,k) = C(k).x(w).Centroid;
                a(:,w,k) = A(k).x(w).Area;

                g = c(1,w,k); % x-coordinate
                l = c(2,w,k); % y-coordinate

                crp = imcrop(img, B(k).x(w).BoundingBox);
                %                 Uncomment to resize images
                %                 zsximgcrp1 = imcrop(img_final, B(i).x(w).BoundingBox);
                %                 sz = size(crp);
                %
                %                 if sz(1) || sz(2) < 224
                %
                %                     if sz(1) < sz(2)
                %
                %                         crp = imresize(crp, [224 NaN]);
                %                         crp1 = imresize(crp1, [224 NaN]);
                %
                %                     else
                %
                %                         crp = imresize(crp, [NaN 224]);
                %                         crp1 = imresize(crp1, [NaN 224]);
                %
                %                     end
                %
                %                 end

                img_name = sprintf('Image%04d',k);
                img_file_name = sprintf('Experiment#1_%s_Video#%d_%s_%d.jpg',...
                    experimental_dirs{i}, j, img_name, w);

                % TODO: index image name, pixel area, centroid value x, centroid
                % value y

                img_info = {img_file_name, a(:,w,k), g, l};
                file_info = cat(1,file_info,img_info);
                imwrite(crp, fullfile(folder, img_file_name));

            end
        end
    end

end
cd(folder);
T = cell2table(file_info,'VariableNames',{'Name' 'Area' 'x' 'y'});
writetable(T,'image_info_neurons.txt');
