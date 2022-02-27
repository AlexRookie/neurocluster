function theta = quaternion2theta(quat)
% Alessandro Antonucci @AlexRookie
% University of Trento

eul = quat2eul([quat.X.Data, quat.Y.Data, quat.Z.Data, quat.W.Data], 'ZYX');
theta = eul(:,3);

end