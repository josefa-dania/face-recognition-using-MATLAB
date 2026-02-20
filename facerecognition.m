clc; clear; close all;
cam = webcam;
faceDetector = vision.CascadeObjectDetector();
net = resnet50; 

dataset_file = 'face_embeddings.mat'; 

disp('Select an option:');
disp('1. Register a new face');
disp('2. Testing Access control');
choice = input('Enter your choice (1 or 2): ');

if choice == 1  
    embeddings_list = [];
    
    
    fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);

    disp('- Registering face details -');
    start_time = tic;

    while toc(start_time) < 10  
        frame = snapshot(cam);
        bbox = step(faceDetector, frame);

        imshow(frame);
        title('Face Registration in Progress', 'FontSize', 18);
        drawnow;

        if ~isempty(bbox)
            face = imcrop(frame, bbox(1, :));  
            face = imresize(face, [224, 224]);  
            face = single(face);  

            embedding = squeeze(activations(net, face, 'fc1000'));  
            embeddings_list = [embeddings_list, embedding(:)];
        end

        pause(0.01);  
    end

    close(fig);  

    if isempty(embeddings_list)
        error('No faces registered! Please try again.');
    end

    registered_embedding = mean(embeddings_list, 2);  
    save(dataset_file, 'registered_embedding');
    disp('Face registration complete!');
end 

if choice == 2  
    if exist(dataset_file, 'file')
        load(dataset_file, 'registered_embedding');
        disp('Face dataset loaded. Starting real-time face recognition...');
    else
        error('No registered faces found! Please register a face first.');
    end

    fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);

    while true
        frame = snapshot(cam);
        bbox = step(faceDetector, frame);

        for i = 1:size(bbox,1)
            face = imcrop(frame, bbox(i, :));
            face = imresize(face, [224, 224]);  
            face = single(face);  

            detected_embedding = squeeze(activations(net, face, 'fc1000'));

            if numel(detected_embedding) ~= numel(registered_embedding)
                disp('Error: Feature size mismatch.');
                continue;
            end

            similarity = dot(registered_embedding(:), detected_embedding(:)) / ...
                        (norm(registered_embedding(:)) * norm(detected_embedding(:)));

            if similarity > 0.9
                frame = insertObjectAnnotation(frame, 'rectangle', bbox(i,:), 'Access Granted', ...
                    'Color', 'green', 'FontSize', 22, 'LineWidth', 3);
            else
                frame = insertObjectAnnotation(frame, 'rectangle', bbox(i,:), 'Access Denied', ...
                    'Color', 'red', 'FontSize', 22, 'LineWidth', 3);
            end
        end

        imshow(frame);
        title('Press "q" to quit', 'FontSize', 18);
        drawnow;

        key = get(gcf, 'CurrentCharacter');
        if any(key == 'q')
            break;
        end
    end
end

clear cam;
close all;
disp('Thank you');
