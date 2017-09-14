

function generate_anchor_scales_ratios(file)
    load(file);
    % stream = RandStream('mlfg6331_64');  % Random number stream
    % options = statset('UseParallel',1,'UseSubstreams',1,...
    %     'Streams',stream);

    [~, C] = kmeans([scaled_height', scaled_width'], 25, 'Display', 'iter', 'Start', 'uniform', 'MaxIter', 1000, 'Replicates', 8);
    figure(1)
    scatter(C(:, 1), C(:, 2))
    xlabel('scaled height')
    ylabel('scaled width')

    D(:, 1) = C(:, 1) ./ C(:, 2);
    D(:, 2) = sqrt(C(:, 1) .* C(:, 2)) / 16;
    figure(2)
    scatter(D(:, 1), D(:, 2))
    xlabel('anchor ratios')
    ylabel('anchor scales')

    for i = 1:1:size(D, 1)
        fprintf('%2.3f, ', D(i, 1));
    end
    fprintf('\n');


    for i = 1:1:size(D, 1)
        fprintf('%2.3f, ', D(i, 2));
    end
    fprintf('\n');