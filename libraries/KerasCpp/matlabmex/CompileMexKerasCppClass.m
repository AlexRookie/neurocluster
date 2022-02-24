clear functions; %#ok<CLFUNC>

name = 'KerasCppClass';

disp('---------------------------------------------------------');
fprintf(1,'Compiling: %s\n',name);

CMD = [ 'while mislocked(''' name '''); munlock(''' name '''); end;'];
eval(CMD);

CMD = [ 'mex -I../src KerasCppClass.cpp -output matlabmex/src/mex_', name ];
CMD = [ CMD, ' -largeArrayDims matlabmex/mex_', name ];
CMD = [ CMD, '.cpp ' ];

if isunix
    if ismac
        CMD = [CMD, ' -lstdc++ CXXFLAGS="\$CXXFLAGS -O2 -g0"'];
    else
        CMD = [CMD, ' -lstdc++ CXXFLAGS="\$CXXFLAGS -Wall -O2 -g0"'];
    end
elseif ispc

end

disp(CMD);
eval(CMD);

clear name CMD;

disp('----------------------- DONE ----------------------------');
