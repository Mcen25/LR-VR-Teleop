using UnityEngine;
using System.Collections.Generic;

public class IKController : MonoBehaviour
{
    [Header("IK Target")]
    public Transform target;
    
    [Header("Robot Configuration")]
    public ArticulationBody rootBody;
    public Transform endEffector;
    
    [Header("IK Settings")]
    [Range(1, 100)]
    public int iterations = 30;
    public float positionThreshold = 0.001f;
    public float orientationThreshold = 1f;
    
    [Header("Orientation Control")]
    public EndEffectorAxis forwardAxis = EndEffectorAxis.NegativeZ;
    public EndEffectorAxis upAxis = EndEffectorAxis.PositiveY;
    
    [Range(0.1f, 2f)]
    public float positionGain = 1.0f;
    
    [Range(0.1f, 2f)]
    public float orientationGain = 1.0f;
    
    [Range(0.01f, 1f)]
    public float dampingFactor = 0.1f;
    
    [Range(1f, 10f)]
    public float baseJointWeight = 3.0f;
    
    [Range(1f, 10f)]
    public float wristRollWeight = 2.0f;
    
    public enum EndEffectorAxis
    {
        PositiveX, NegativeX,
        PositiveY, NegativeY,
        PositiveZ, NegativeZ
    }
    
    [Header("Workspace Limits")]
    public float maxReach = 0.35f;
    public float minReach = 0.05f;
    public float minHeight = -0.1f;
    
    [Header("Smoothing")]
    [Range(0.01f, 1f)]
    public float angleLerpSpeed = 0.5f;
    public float maxAngleChangePerStep = 10f;
    
    [Range(0f, 0.95f)]
    public float solutionSmoothing = 0.7f;
    
    [Header("Joint Drive Settings")]
    public float driveStiffness = 100000f;
    public float driveDamping = 10000f;
    public float driveForceLimit = 10000f;
    
    [Header("Debug")]
    public bool showDebugGizmos = true;
    public bool showWorkspaceLimits = true;
    public bool showOrientationAxes = true;
    public bool showJointAxes = true;
    
    private List<JointInfo> jointInfos = new List<JointInfo>();
    
    private float[] targetAngles;
    private float[] currentTargetAngles;
    private float[] solverAngles;
    private float[] smoothedSolverAngles;
    
    private Vector3 clampedTargetPosition;
    private Quaternion clampedTargetRotation;
    private bool targetInRange;
    
    private const int TaskSpaceDim = 6;
    private float[,] jacobian;
    private float[] errorVector;
    private float[] deltaAngles;
    private float[] jointWeights;
    
    private struct JointInfo
    {
        public ArticulationBody body;
        public Vector3 localAxis;
        public float lowerLimit;
        public float upperLimit;
    }
    
    void Start()
    {
        if (rootBody == null || endEffector == null)
        {
            Debug.LogError("IKController: Please assign rootBody and endEffector!");
            enabled = false;
            return;
        }
        
        BuildJointChain();
        ConfigureJointDrives();
        
        if (maxReach <= 0) EstimateWorkspaceLimits();
        
        int n = jointInfos.Count;
        targetAngles = new float[n];
        currentTargetAngles = new float[n];
        solverAngles = new float[n];
        smoothedSolverAngles = new float[n];
        
        jacobian = new float[TaskSpaceDim, n];
        errorVector = new float[TaskSpaceDim];
        deltaAngles = new float[n];
        jointWeights = new float[n];
        
        for (int i = 0; i < n; i++)
        {
            jointWeights[i] = 1.0f;
        }
        if (n > 0) jointWeights[0] = baseJointWeight;
        if (n > 0) jointWeights[n - 1] = wristRollWeight;
        
        for (int i = 0; i < n; i++)
        {
            float angle = GetJointAngle(jointInfos[i].body);
            targetAngles[i] = angle;
            currentTargetAngles[i] = angle;
            solverAngles[i] = angle;
            smoothedSolverAngles[i] = angle;
        }
    }
    
    void EstimateWorkspaceLimits()
    {
        float totalLength = 0f;
        for (int i = 0; i < jointInfos.Count - 1; i++)
        {
            totalLength += Vector3.Distance(
                jointInfos[i].body.transform.position,
                jointInfos[i + 1].body.transform.position
            );
        }
        if (jointInfos.Count > 0)
        {
            totalLength += Vector3.Distance(
                jointInfos[jointInfos.Count - 1].body.transform.position,
                endEffector.position
            );
        }
        maxReach = totalLength * 0.95f;
        minReach = totalLength * 0.1f;
        Debug.Log($"IKController: Estimated max reach = {maxReach:F3}m");
    }
    
    void BuildJointChain()
    {
        jointInfos.Clear();
        BuildJointChainFromHierarchy(rootBody.transform);
        
        Debug.Log($"IKController: Built joint chain with {jointInfos.Count} joints");
        foreach (var info in jointInfos)
        {
            Debug.Log($"  - {info.body.name}: limits [{info.lowerLimit * Mathf.Rad2Deg:F1}°, {info.upperLimit * Mathf.Rad2Deg:F1}°]");
        }
    }
    
    void BuildJointChainFromHierarchy(Transform current)
    {
        var body = current.GetComponent<ArticulationBody>();
        if (body != null && body.jointType == ArticulationJointType.RevoluteJoint)
        {
            if (!current.name.ToLower().Contains("jaw"))
            {
                jointInfos.Add(new JointInfo
                {
                    body = body,
                    localAxis = body.anchorRotation * Vector3.right,
                    lowerLimit = body.xDrive.lowerLimit * Mathf.Deg2Rad,
                    upperLimit = body.xDrive.upperLimit * Mathf.Deg2Rad
                });
            }
        }
        
        if (current == endEffector || current == endEffector.parent) return;
        foreach (Transform child in current)
            BuildJointChainFromHierarchy(child);
    }
    
    void ConfigureJointDrives()
    {
        foreach (var info in jointInfos)
        {
            var drive = info.body.xDrive;
            drive.stiffness = driveStiffness;
            drive.damping = driveDamping;
            drive.forceLimit = driveForceLimit;
            info.body.xDrive = drive;
        }
    }
    
    Vector3 GetAxisVector(EndEffectorAxis axis)
    {
        switch (axis)
        {
            case EndEffectorAxis.PositiveX: return Vector3.right;
            case EndEffectorAxis.NegativeX: return Vector3.left;
            case EndEffectorAxis.PositiveY: return Vector3.up;
            case EndEffectorAxis.NegativeY: return Vector3.down;
            case EndEffectorAxis.PositiveZ: return Vector3.forward;
            case EndEffectorAxis.NegativeZ: return Vector3.back;
            default: return Vector3.forward;
        }
    }
    
    Vector3 ClampTargetToWorkspace(Vector3 targetPos)
    {
        Vector3 basePos = rootBody.transform.position;
        Vector3 toTarget = targetPos - basePos;
        float distance = toTarget.magnitude;
        
        targetInRange = true;
        Vector3 clamped = targetPos;
        
        if (distance > maxReach)
        {
            clamped = basePos + toTarget.normalized * maxReach;
            targetInRange = false;
        }
        else if (distance < minReach)
        {
            clamped = basePos + toTarget.normalized * minReach;
            targetInRange = false;
        }
        
        if (clamped.y < basePos.y + minHeight)
        {
            clamped.y = basePos.y + minHeight;
            targetInRange = false;
        }
        
        return clamped;
    }
    
    void FixedUpdate()
    {
        if (target == null) return;
        
        clampedTargetPosition = ClampTargetToWorkspace(target.position);
        clampedTargetRotation = target.rotation;
        
        System.Array.Copy(smoothedSolverAngles, solverAngles, jointInfos.Count);
        
        SolveIKJacobian();
        
        for (int i = 0; i < jointInfos.Count; i++)
        {
            smoothedSolverAngles[i] = Mathf.Lerp(solverAngles[i], smoothedSolverAngles[i], solutionSmoothing);
            smoothedSolverAngles[i] = Mathf.Clamp(smoothedSolverAngles[i], jointInfos[i].lowerLimit, jointInfos[i].upperLimit);
            targetAngles[i] = smoothedSolverAngles[i];
        }
        
        ApplySmoothedAngles();
    }
    
    void SolveIKJacobian()
    {
        int n = jointInfos.Count;
        
        for (int iter = 0; iter < iterations; iter++)
        {
            Vector3 eePos = GetSimulatedEndEffectorPosition();
            Quaternion eeRot = GetSimulatedEndEffectorRotation();
            
            Vector3 posError = clampedTargetPosition - eePos;
            Vector3 rotError = CalculateOrientationError(eeRot, clampedTargetRotation);
            
            float posMag = posError.magnitude;
            float rotMag = rotError.magnitude * Mathf.Rad2Deg;
            
            if (posMag < positionThreshold && rotMag < orientationThreshold)
                break;
            
            posError *= positionGain;
            rotError *= orientationGain;
            
            errorVector[0] = posError.x;
            errorVector[1] = posError.y;
            errorVector[2] = posError.z;
            errorVector[3] = rotError.x;
            errorVector[4] = rotError.y;
            errorVector[5] = rotError.z;
            
            ComputeJacobian(eePos, eeRot);
            
            SolveDampedLeastSquares();
            
            for (int i = 0; i < n; i++)
            {
                float newAngle = solverAngles[i] + deltaAngles[i];
                
                float margin = 0.15f;
                float distToLower = newAngle - jointInfos[i].lowerLimit;
                float distToUpper = jointInfos[i].upperLimit - newAngle;
                
                if (distToLower < margin && deltaAngles[i] < 0)
                    newAngle = solverAngles[i] + deltaAngles[i] * Mathf.Clamp01(distToLower / margin);
                else if (distToUpper < margin && deltaAngles[i] > 0)
                    newAngle = solverAngles[i] + deltaAngles[i] * Mathf.Clamp01(distToUpper / margin);
                
                solverAngles[i] = Mathf.Clamp(newAngle, jointInfos[i].lowerLimit, jointInfos[i].upperLimit);
            }
        }
    }
    
    void ComputeJacobian(Vector3 eePos, Quaternion eeRot)
    {
        int n = jointInfos.Count;
        
        Vector3 targetUp = clampedTargetRotation * Vector3.up;
        
        for (int i = 0; i < n; i++)
        {
            var info = jointInfos[i];
            Vector3 jointPos = info.body.transform.position;
            
            Vector3 axis = info.body.transform.TransformDirection(info.localAxis).normalized;
            
            Vector3 r = eePos - jointPos;
            Vector3 linearCol = Vector3.Cross(axis, r);
            Vector3 angularCol = axis;
            
            float weight = jointWeights[i];
            
            if (i == 0)
            {
                Vector3 basePos = jointInfos[0].body.transform.position;
                Vector3 toTarget = clampedTargetPosition - basePos;
                toTarget.y = 0;
                
                if (toTarget.magnitude > 0.01f)
                {
                    Vector3 tangent = Vector3.Cross(Vector3.up, toTarget.normalized);
                    linearCol += tangent * weight * 0.5f;
                }
            }
            
            if (i == n - 1)
            {
                Vector3 currentUp = eeRot * Vector3.up;
                
                float yContribution = Mathf.Abs(Vector3.Dot(axis, Vector3.up));
                if (yContribution < 0.9f)
                {
                    angularCol *= weight;
                }
            }
            
            jacobian[0, i] = linearCol.x * weight;
            jacobian[1, i] = linearCol.y * weight;
            jacobian[2, i] = linearCol.z * weight;
            jacobian[3, i] = angularCol.x * weight;
            jacobian[4, i] = angularCol.y * weight;
            jacobian[5, i] = angularCol.z * weight;
        }
    }
    
    void SolveDampedLeastSquares()
    {
        int n = jointInfos.Count;
        float lambda2 = dampingFactor * dampingFactor;
        
        float[,] JJt = new float[TaskSpaceDim, TaskSpaceDim];
        for (int i = 0; i < TaskSpaceDim; i++)
        {
            for (int j = 0; j < TaskSpaceDim; j++)
            {
                float sum = 0;
                for (int k = 0; k < n; k++)
                    sum += jacobian[i, k] * jacobian[j, k];
                JJt[i, j] = sum;
                
                if (i == j) JJt[i, j] += lambda2;
            }
        }
        
        float[] y = SolveLinearSystem(JJt, errorVector, TaskSpaceDim);
        
        for (int i = 0; i < n; i++)
        {
            float sum = 0;
            for (int j = 0; j < TaskSpaceDim; j++)
                sum += jacobian[j, i] * y[j];
            deltaAngles[i] = sum;
        }
    }
    
    float[] SolveLinearSystem(float[,] A, float[] b, int size)
    {
        float[,] M = new float[size, size];
        float[] x = new float[size];
        System.Array.Copy(b, x, size);
        
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                M[i, j] = A[i, j];
        
        for (int col = 0; col < size; col++)
        {
            int maxRow = col;
            float maxVal = Mathf.Abs(M[col, col]);
            for (int row = col + 1; row < size; row++)
            {
                if (Mathf.Abs(M[row, col]) > maxVal)
                {
                    maxVal = Mathf.Abs(M[row, col]);
                    maxRow = row;
                }
            }
            
            if (maxRow != col)
            {
                for (int j = 0; j < size; j++)
                {
                    float tmp = M[col, j];
                    M[col, j] = M[maxRow, j];
                    M[maxRow, j] = tmp;
                }
                float tmpB = x[col];
                x[col] = x[maxRow];
                x[maxRow] = tmpB;
            }
            
            float pivot = M[col, col];
            if (Mathf.Abs(pivot) < 1e-10f) continue;
            
            for (int row = col + 1; row < size; row++)
            {
                float factor = M[row, col] / pivot;
                for (int j = col; j < size; j++)
                    M[row, j] -= factor * M[col, j];
                x[row] -= factor * x[col];
            }
        }
        
        for (int i = size - 1; i >= 0; i--)
        {
            float sum = x[i];
            for (int j = i + 1; j < size; j++)
                sum -= M[i, j] * x[j];
            x[i] = Mathf.Abs(M[i, i]) > 1e-10f ? sum / M[i, i] : 0;
        }
        
        return x;
    }
    
    Vector3 CalculateOrientationError(Quaternion current, Quaternion target)
    {
        Quaternion errorQuat = target * Quaternion.Inverse(current);
        
        if (errorQuat.w < 0)
        {
            errorQuat = new Quaternion(-errorQuat.x, -errorQuat.y, -errorQuat.z, -errorQuat.w);
        }
        
        errorQuat.ToAngleAxis(out float angle, out Vector3 axis);
        
        if (angle > 180f)
        {
            angle = 360f - angle;
            axis = -axis;
        }
        
        if (angle < 0.01f || float.IsNaN(axis.x))
        {
            return Vector3.zero;
        }
        
        Vector3 errorVec = axis.normalized * (angle * Mathf.Deg2Rad);
        
        errorVec.y *= wristRollWeight;
        
        return errorVec;
    }
    
    Vector3 GetSimulatedEndEffectorPosition()
    {
        Vector3 eePos = endEffector.position;
        
        for (int i = 0; i < jointInfos.Count; i++)
        {
            var info = jointInfos[i];
            float currentAngle = GetJointAngle(info.body);
            float deltaAngle = solverAngles[i] - currentAngle;
            
            if (Mathf.Abs(deltaAngle) > 0.0001f)
            {
                Vector3 jointPos = info.body.transform.position;
                Vector3 worldAxis = info.body.transform.TransformDirection(info.localAxis);
                Quaternion rotation = Quaternion.AngleAxis(deltaAngle * Mathf.Rad2Deg, worldAxis);
                eePos = jointPos + rotation * (eePos - jointPos);
            }
        }
        
        return eePos;
    }
    
    Quaternion GetSimulatedEndEffectorRotation()
    {
        Quaternion eeRot = endEffector.rotation;
        
        for (int i = 0; i < jointInfos.Count; i++)
        {
            var info = jointInfos[i];
            float currentAngle = GetJointAngle(info.body);
            float deltaAngle = solverAngles[i] - currentAngle;
            
            if (Mathf.Abs(deltaAngle) > 0.0001f)
            {
                Vector3 worldAxis = info.body.transform.TransformDirection(info.localAxis);
                Quaternion rotation = Quaternion.AngleAxis(deltaAngle * Mathf.Rad2Deg, worldAxis);
                eeRot = rotation * eeRot;
            }
        }
        
        return eeRot;
    }
    
    void ApplySmoothedAngles()
    {
        for (int i = 0; i < jointInfos.Count; i++)
        {
            float angleDiff = targetAngles[i] - currentTargetAngles[i];
            float maxChange = maxAngleChangePerStep * Mathf.Deg2Rad;
            angleDiff = Mathf.Clamp(angleDiff, -maxChange, maxChange);
            
            currentTargetAngles[i] += angleDiff * angleLerpSpeed;
            currentTargetAngles[i] = Mathf.Clamp(currentTargetAngles[i], jointInfos[i].lowerLimit, jointInfos[i].upperLimit);
            
            var drive = jointInfos[i].body.xDrive;
            drive.target = currentTargetAngles[i] * Mathf.Rad2Deg;
            jointInfos[i].body.xDrive = drive;
        }
    }
    
    float GetJointAngle(ArticulationBody joint)
    {
        return joint.jointPosition.dofCount > 0 ? joint.jointPosition[0] : 0f;
    }
    
    public void SetJointPositionsDirect(float[] anglesRadians)
    {
        if (anglesRadians.Length != jointInfos.Count) return;
        
        rootBody.immovable = true;
        for (int i = 0; i < jointInfos.Count; i++)
        {
            var joint = jointInfos[i].body;
            float angle = Mathf.Clamp(anglesRadians[i], jointInfos[i].lowerLimit, jointInfos[i].upperLimit);
            
            joint.jointPosition = new ArticulationReducedSpace(angle);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
            
            currentTargetAngles[i] = angle;
            targetAngles[i] = angle;
            solverAngles[i] = angle;
            smoothedSolverAngles[i] = angle;
            
            var drive = joint.xDrive;
            drive.target = angle * Mathf.Rad2Deg;
            joint.xDrive = drive;
        }
        rootBody.immovable = false;
    }
    
    public void ResetToHome() => SetJointPositionsDirect(new float[jointInfos.Count]);
    
    public float GetDistanceToTarget() => endEffector != null && target != null ? 
        Vector3.Distance(endEffector.position, target.position) : float.MaxValue;
    
    public float GetOrientationError() => endEffector != null && target != null ? 
        Quaternion.Angle(endEffector.rotation, target.rotation) : float.MaxValue;
    
    public bool IsTargetReachable => targetInRange;
    public Vector3 ClampedTargetPosition => clampedTargetPosition;
    public int JointCount => jointInfos.Count;
    
    void OnDrawGizmos()
    {
        if (!showDebugGizmos) return;
        
        if (target != null)
        {
            Gizmos.color = targetInRange ? Color.green : Color.red;
            Gizmos.DrawWireSphere(target.position, 0.015f);
            
            if (!targetInRange)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(clampedTargetPosition, 0.02f);
                Gizmos.DrawLine(target.position, clampedTargetPosition);
            }
            
            if (showOrientationAxes)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawRay(target.position, target.rotation * GetAxisVector(forwardAxis) * 0.05f);
                Gizmos.color = Color.green;
                Gizmos.DrawRay(target.position, target.rotation * GetAxisVector(upAxis) * 0.04f);
                Gizmos.color = Color.blue;
                Gizmos.DrawRay(target.position, target.rotation * Vector3.Cross(GetAxisVector(upAxis), GetAxisVector(forwardAxis)) * 0.03f);
            }
        }
        
        if (endEffector != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(endEffector.position, 0.012f);
            
            if (showOrientationAxes)
            {
                Gizmos.color = new Color(1f, 0.5f, 0.5f);
                Gizmos.DrawRay(endEffector.position, endEffector.rotation * GetAxisVector(forwardAxis) * 0.05f);
                Gizmos.color = new Color(0.5f, 1f, 0.5f);
                Gizmos.DrawRay(endEffector.position, endEffector.rotation * GetAxisVector(upAxis) * 0.04f);
                Gizmos.color = new Color(0.5f, 0.5f, 1f);
                Gizmos.DrawRay(endEffector.position, endEffector.rotation * Vector3.Cross(GetAxisVector(upAxis), GetAxisVector(forwardAxis)) * 0.03f);
            }
        }
        
        if (target != null && endEffector != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(endEffector.position, clampedTargetPosition);
        }
        
        if (showWorkspaceLimits && rootBody != null)
        {
            Vector3 basePos = rootBody.transform.position;
            Gizmos.color = new Color(0, 1, 0, 0.1f);
            Gizmos.DrawWireSphere(basePos, maxReach);
            Gizmos.color = new Color(1, 0, 0, 0.1f);
            Gizmos.DrawWireSphere(basePos, minReach);
        }
        
        if (showJointAxes && jointInfos != null && jointInfos.Count > 0)
        {
            for (int i = 0; i < jointInfos.Count; i++)
            {
                var info = jointInfos[i];
                if (info.body == null) continue;
                
                Vector3 jointPos = info.body.transform.position;
                Vector3 worldAxis = info.body.transform.TransformDirection(info.localAxis).normalized;
                
                if (i == 0)
                    Gizmos.color = Color.magenta;
                else if (i == jointInfos.Count - 1)
                    Gizmos.color = Color.yellow;
                else
                    Gizmos.color = Color.cyan;
                
                Gizmos.DrawRay(jointPos, worldAxis * 0.05f);
                Gizmos.DrawWireSphere(jointPos + worldAxis * 0.05f, 0.005f);
            }
        }
    }
}