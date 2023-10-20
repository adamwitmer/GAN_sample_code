close all; clear all; clc;

%%% this script is designed to assist in the manual ground truth labeling
%%% of image data for the use in machine learning/deep neural network
%%% training. the program takes a path to the image data as an input, and
%%% then shows the images to a user 1-by-1, taking the suggestion of the
%%% user as a ground truth label, and outputting a csv file contatining
%%% image names and their corresponding user-defined labels. 

% define image classes and initialize label array
classes = {'Debris', 'Dense', 'Diff', 'Spread', 'Unsure'};

% input file path to image folder here
imgfolder = '/data3/adamw/AUXGAN_save/example_dataset';
cd(imgfolder);

% make separate folders for each class within 'imgfolder'
warning off; 
for imgclass = 1:length(classes)
    mkdir(classes{imgclass})
end

% list all images within imgfolder (yet to be labelled)
imglist = dir('*.jpg');

% begin labeling images one-by-one
for j = 1:length(imglist)

    close all; clc;
    fprintf('Showing image %d/%d', j, length(imglist));
    imgname = imglist(j).name;
    
    if exist(imgname)
        img = imread(imgname);
        fig = figure; movegui(fig,'east'); imshow(img)
        prompt = ['Label: Enter 1 - Debris, 2 - Dense, 3 - Differentiated,'...
            '4 - Spread, 5 - Unsure, or type ''back'' to see previous image'...
            ' or ''quit'' to exit program'];
        warning off;
        inpt = input(prompt, 's');

        try 
            if inpt == '1'
                movefile(imgname, fullfile(imgfolder, classes{1}));
            elseif inpt == '2'
                movefile(imgname, fullfile(imgfolder, classes{2}));
            elseif inpt == '3'
                movefile(imgname, fullfile(imgfolder, classes{3}));
            elseif inpt == '4'
                movefile(imgname, fullfile(imgfolder, classes{4}));
            elseif inpt == '5'
                movefile(imgname, fullfile(imgfolder, classes{5}));
            elseif strcmpi(inpt, 'back') && j > 1 
                
                % if input == 'back', load previous picture from
                % new directory and take input again and move file
                close all; clc;
                lastimg = imread(fullfile(imgfolder, classes{str2double(lastlabel)},...
                    lastname));
                fig = figure; movegui(fig, 'east'); imshow(lastimg);
                prompt = 'Please re-enter desired input...';
                inpt = input(prompt, 's');
                if ismember(inpt, {'1', '2', '3', '4', '5'})
                    if inpt ==  lastlabel
                        disp('Error: label entered same as previous, please enter new label or hit enter to continue')
                        inpt = input(prompt, 's');
                        if ismember(inpt, {'1', '2', '3', '4', '5'})
                            movefile(fullfile(imgfolder, classes{str2double(lastlabel)},...
                                    lastname),fullfile(imgfolder, classes{str2double(inpt)},...
                                    lastname));
                        
                            lastlabel = inpt;
                        end
                         % after re-entry, show and label second image 
                        close all; clc;
                        fig = figure; movegui(fig, 'east'); imshow(img);
                        prompt = ['Label: Enter 1 - Debris, 2 - Dense, 3 - Differentiated,'...
                                    '4 - Spread, 5 - Unsure, or type ''back'' to see previous image'...
                                    ' or ''quit'' to exit program'];               
                        inpt = input(prompt, 's');
                        if ismember(inpt, {'1', '2', '3', '4', '5'})
                            movefile(imgname, fullfile(imgfolder,...
                                classes{str2double(inpt)}));
                            lastname = imgname;
                            lastlabel = inpt;
                            continue
                        elseif isempty(inpt)
                            continue
                        elseif strcmpi(inpt, 'quit')
                            disp('quitting program');
                            return
                        else
                            disp('Error occurred, quitting program, please restart to continue')
                            return
                        end
                    end
                    movefile(fullfile(imgfolder, classes{str2double(lastlabel)},...
                        lastname),fullfile(imgfolder, classes{str2double(inpt)},...
                        lastname));
                    lastlabel = inpt;
                elseif isempty(inpt)
                    continue
                elseif strcmpi(inpt, 'quit')
                    disp('quitting program');
                    return 
                else 
                    
                    return
                end
                
                % after re-entry, show and label second image 
                close all; clc;
                fig = figure; movegui(fig, 'east'); imshow(img);
                prompt = ['Label: Enter 1 - Debris, 2 - Dense, 3 - Differentiated,'...
                            '4 - Spread, 5 - Unsure, or type ''back'' to see previous image'...
                            ' or ''quit'' to exit program'];               
                inpt = input(prompt, 's');
                if ismember(inpt, {'1', '2', '3', '4', '5'})
                    movefile(imgname, fullfile(imgfolder,...
                        classes{str2double(inpt)}));
                    lastname = imgname;
                    lastlabel = inpt;
                    continue
                elseif isempty(inpt)
                    continue
                elseif strcmpi(inpt, 'quit')
                    disp('quitting program');
                    return
                else
                    disp('Error occurred, quitting program, please restart to continue')
                    return
                end
            elseif isempty(inpt)
                continue
            elseif strcmpi(inpt, 'quit')
                disp('quitting program');
                return
            else 
                disp('Error occurred, quitting program, please restart to continue')
                return
            end
            lastname = imgname;
            lastlabel = inpt;
        catch 
            prompt = ('Incorrect input, please re-enter...');
            inpt = input(prompt,'s');
            if inpt == '1'
                movefile(imgname, fullfile(imgfolder, classes{1}));
            elseif inpt == '2'
                movefile(imgname, fullfile(imgfolder, classes{2}));
            elseif inpt == '3'
                movefile(imgname, fullfile(imgfolder, classes{3}));
            elseif inpt == '4'
                movefile(imgname, fullfile(imgfolder, classes{4}));
            elseif inpt == '5'
                movefile(imgname, fullfile(imgfolder, classes{5}));
            elseif strcmpi(inpt, 'back') && j > 1 
                
                % if input == 'back', load previous picture from
                % new directory and take input again and move file
                close all; clc;
                lastimg = imread(fullfile(imgfolder, classes{str2double(lastlabel)},...
                    lastname));
                fig = figure; movegui(fig, 'east'); imshow(lastimg)
                prompt = 'Please re-enter desired input...';
                inpt = input(prompt, 's');
                if ismember(inpt, {'1', '2', '3', '4', '5'})
                    if inpt ==  lastlabel
                        disp('Error: label entered same as previous, please enter new label or hit enter to continue')
                        inpt = input(prompt, 's');
                        if ismember(inpt, {'1', '2', '3', '4', '5'})
                            movefile(fullfile(imgfolder, classes{str2double(lastlabel)},...
                                    lastname),fullfile(imgfolder, classes{str2double(inpt)},...
                                    lastname));
                        
                            lastlabel = inpt;
                        end
                         % after re-entry, show and label second image 
                        close all; clc;
                        fig = figure; movegui(fig, 'east'); imshow(img);
                        prompt = ['Label: Enter 1 - Debris, 2 - Dense, 3 - Differentiated,'...
                                    '4 - Spread, 5 - Unsure, or type ''back'' to see previous image'...
                                    ' or ''quit'' to exit program'];               
                        inpt = input(prompt, 's');
                        if ismember(inpt, {'1', '2', '3', '4', '5'})
                            movefile(imgname, fullfile(imgfolder,...
                                classes{str2double(inpt)}));
                            lastname = imgname;
                            lastlabel = inpt;
                            continue
                        elseif isempty(inpt)
                            continue
                        elseif strcmpi(inpt, 'quit')
                            disp('quitting program');
                            return
                        else
                            disp('Error occurred, quitting program, please restart to continue')
                            return
                        end
                    end
                    movefile(fullfile(imgfolder, classes{str2double(lastlabel)},...
                        lastname),fullfile(imgfolder, classes{str2double(inpt)},...
                        lastname));
                    lastlabel = inpt;
                elseif isempty(inpt)
                    continue
                elseif strcmpi(inpt, 'quit')
                    disp('quitting program');
                    return
                else
                    disp('Error occurred, quitting program, please restart to continue')
                    return
                end
                
                % after re-entry, show and label second image 
                close all; clc;
                imgname = imglist(j).name;
                img = imread(imgname);
                fig = figure; movegui(fig, 'east'); imshow(img);
                prompt = ['Label: Enter 1 - Debris, 2 - Dense, 3 - Differentiated,'...
                            '4 - Spread, 5 - Unsure, or type ''back'' to see previous image'...
                            ' or ''quit'' to exit program'];
                inpt = input(prompt, 's');
                if ismember(inpt, {'1', '2', '3', '4', '5'})
                    movefile(imgname, fullfile(imgfolder,...
                        classes{str2double(inpt)}));
                    lastname = imgname;
                    lastlabel = inpt;
                    continue
                elseif isempty(inpt)
                    continue
                elseif strcmpi(inpt, 'quit')
                    disp('quitting program');
                    return
                else
                    disp('Error occurred, quitting program, please restart to continue')
                    return
                end
                
            elseif isempty(inpt)
                continue
            elseif strcmpi(inpt, 'quit')
                disp('quitting program');
                return
            else
                disp('Error occurred, quitting program, please restart to continue')
                return
            end
            lastname = imgname;
            lastlabel = inpt;  
        end
            
    end 
        
end