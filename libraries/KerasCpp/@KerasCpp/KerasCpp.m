classdef KerasCpp < handle
    
    properties (SetAccess = protected, Hidden = true)
        objectHandle; % Store the pointer of the C++ class
    end
    
    methods
        
        %==== CPP METHODS =================================================
        
        % Constructor: initialize the state
        function self = KerasCpp()
            self.objectHandle = mex_KerasCppClass('new');
        end
        
        % Destructor
        function delete(self)
            % Destroy the C++ class instance
            if self.objectHandle ~= 0
                mex_KerasCppClass('delete', self.objectHandle);
            end
            self.objectHandle = 0; % avoid double destruction of object
        end
        
        % Inference with the trained network
        function outputs = predict(self, inputs)
            outputs = mex_KerasCppClass('predict', self.objectHandle, inputs);
        end
        
        %==== MATLAB METHODS ==============================================
        
    end % end of public methods
    
    methods (Access = private)
        
    end % end of private methods
    
end
