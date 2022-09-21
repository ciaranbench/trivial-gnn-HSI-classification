
close all
clear

load('fat_spec.mat')
load('muscle_spec.mat')


muscle_spec = muscle_spec(2,1:100);
muscle_spec = muscle_spec./max((fat_spec(:)));
fat_spec = fat_spec(2,1:100);
fat_spec = fat_spec./max((fat_spec(:)));




muscle_spec = muscle_spec';

fat_spec = fat_spec';


multi_body = imread("multi_body.PNG");
multi_body  = double(im2bw(imresize(multi_body,[100,100])));

multi_body(find(multi_body==0))=2;

multi_body(find(multi_body==1))=0;

multi_body(find(multi_body==2))=1;


single_body = imread("single_body.PNG");
single_body  =  double(im2bw(imresize(single_body,[100,100])));
single_body(find(single_body==0))=2;
single_body(find(single_body==1))=0;
single_body(find(single_body==2))=1;

figure
imagesc(squeeze(single_body))
figure
imagesc(squeeze(multi_body))
for examples = 1:1:400%300
coin_flip = rand(1)
extra_val = randi(20)-1;
angle = randi(360)-1;
flip_horiz = round(rand(1));
    if coin_flip < 0.5
%add rows and clums, and then resize down to original szie

extra_columns = zeros(size(single_body,1),extra_val);

single_body_large = [extra_columns, single_body, extra_columns];

extra_rows = zeros(extra_val,size(single_body_large,2));
single_body_large = [extra_rows; single_body_large; extra_rows];

single_body_final = im2bw(imresize(single_body_large, [100,100]));
single_body_final = imrotate(single_body_final,angle);

single_body_final = single_body_final(((size(single_body_final,1)/2)-49):((size(single_body_final,1)/2)+50),((size(single_body_final,1)/2)-49):((size(single_body_final,1)/2)+50));

if flip_horiz==1
single_body_final = fliplr(single_body_final);
end

fill = 0.3;
ncols=100;
nrows=100;
[xi,yi] = meshgrid(1:ncols,1:nrows);
a = 1 + 0*rand(1); % Try varying the amplitude of the cubic term.
xt = xi - ncols/2;
yt = yi - nrows/2;
[theta,r] = cart2pol(xt,yt);
rmax = max(r(:));
s1 = r + r.^3*(a/rmax.^2);
b = 0.4 + 0*rand(1); % Try varying the amplitude of the cubic term.
s = r - r.^3*(b/rmax.^2);
[ut,vt] = pol2cart(theta,s);
ui = ut + ncols/2;
vi = vt + nrows/2;
ifcn = @(c) [ui(:) vi(:)];
tform = geometricTransform2d(ifcn);
single_body_final = imwarp(single_body_final,tform,'FillValues',fill);
%figure
%imagesc(squeeze(G_final(:,:)))



%%%%%%%%%%%
fat_coords = find(single_body_final == 1);
[fat_coords_r, fat_coords_c]  = ind2sub(size(single_body_final),fat_coords);
muscle_coords = find(single_body_final == 0);
[muscle_coords_r, muscle_coords_c] = ind2sub(size(single_body_final),muscle_coords);

sample_HSI = zeros(100,100,100);
for i=1:1:length(muscle_coords_r)
sample_HSI(muscle_coords_r(i), muscle_coords_c(i),:) = muscle_spec;
i

end

for i=1:1:length(fat_coords_r)
sample_HSI(fat_coords_r(i), fat_coords_c(i),:) = fat_spec;
i
end
%figure
%imagesc(squeeze(sample_HSI(:,:,70)))

%%
%figure
%imagesc(squeeze(sample_HSI(:,:,25)))
%%
%%%%%%%%%%%



counter = 0;
for i = 10:10:100
    for j = 10:10:100
        counter= counter+1;
        single_body_decomp(counter,:,:,:)=sample_HSI(i-9:i,j-9:j,:);
    end
end
truths(examples,:)=[0,1];
train_set(examples,:,:,:,:)=single_body_decomp;
end
if coin_flip >= 0.5
%add rows and columns, and then resize down to original szie
extra_columns = zeros(size(multi_body,1),extra_val);

multi_body_large = [extra_columns, multi_body, extra_columns];

extra_rows = zeros(extra_val,size(multi_body_large,2));
multi_body_large = [extra_rows; multi_body_large; extra_rows];

multi_body_final = im2bw(imresize(multi_body_large, [100,100]));

multi_body_final = imrotate(multi_body_final,angle);
multi_body_final = multi_body_final(((size(multi_body_final,1)/2)-49):((size(multi_body_final,1)/2)+50),((size(multi_body_final,1)/2)-49):((size(multi_body_final,1)/2)+50));



if flip_horiz==1
multi_body_final = fliplr(multi_body_final);
end


fill = 0.3;
ncols=100;
nrows=100;
[xi,yi] = meshgrid(1:ncols,1:nrows);
a = 1 + 0*rand(1); % Try varying the amplitude of the cubic term.
xt = xi - ncols/2;
yt = yi - nrows/2;
[theta,r] = cart2pol(xt,yt);
rmax = max(r(:));
s1 = r + r.^3*(a/rmax.^2);
b = 0.4 + 0*rand(1); % Try varying the amplitude of the cubic term.
s = r - r.^3*(b/rmax.^2);
[ut,vt] = pol2cart(theta,s);
ui = ut + ncols/2;
vi = vt + nrows/2;
ifcn = @(c) [ui(:) vi(:)];
tform = geometricTransform2d(ifcn);
multi_body_final = imwarp(multi_body_final,tform,'FillValues',fill);
%figure
%imagesc(squeeze(seven_final(:,:)))

%%%%
fat_coords = find(multi_body_final == 1);
[fat_coords_r, fat_coords_c]  = ind2sub(size(multi_body_final),fat_coords);
muscle_coords = find(multi_body_final == 0);
[muscle_coords_r, muscle_coords_c] = ind2sub(size(multi_body_final),muscle_coords);

sample_HSI = zeros(100,100,100);
for i=1:1:length(muscle_coords_r)
sample_HSI(muscle_coords_r(i), muscle_coords_c(i),:) = muscle_spec;
i
end

for i=1:1:length(fat_coords_r)
sample_HSI(fat_coords_r(i), fat_coords_c(i),:) = fat_spec;
i
end

%%
%figure
%imagesc(squeeze(sample_HSI(:,:,25)))
%%

%%%%


counter = 0;
for i = 10:10:100
    for j = 10:10:100
        counter= counter+1;
        multi_body_decomp(counter,:,:,:)=sample_HSI(i-9:i,j-9:j,:);
    end
end

train_set(examples,:,:,:,:)=multi_body_decomp;
truths(examples,:)=[1, 0];
end
end

test_set_parsed = train_set(end-300:end,:,:,:,:);
test_truths = truths(end-300:end,:);


vali_set_examples = train_set(end-340:end-300,:,:,:,:);
counter=1;
for i =1:1:size(vali_set_examples,1)
    for j =1:1:size(vali_set_examples,2)
        vali_set_parsed(counter,:,:,:) = vali_set_examples(i,j,:,:,:);
        counter = counter+1;
    end
end
vali_truths = truths(end-340:end-300,:);


train_set_examples = train_set(1:end-340,:,:,:,:);
counter=1;
for i =1:1:size(train_set_examples,1)
    for j =1:1:size(train_set_examples,2)
        train_set_parsed(counter,:,:,:) = train_set_examples(i,j,:,:,:);

        counter = counter+1;
    end
end
train_truths= truths(1:end-340,:);
save('train_set_parsed.mat','train_set_parsed','-v7.3')
save('vali_set_parsed.mat','vali_set_parsed','-v7.3')
save('test_set_parsed.mat','test_set_parsed','-v7.3')

save('train_truths.mat','train_truths')
save('vali_truths.mat','vali_truths')
save('test_truths.mat','test_truths')

%whole_HSIs = 
%save('whole_HSIs','whole_HSIs','-v7.3')
