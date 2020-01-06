# Liferay Partner List
SELECT Account_.Name as "Account Name"
		,Opportunity.Name as "Opportunity Name"
		,Opportunity.Partnership_Start_Date__c as "Partnership Start Date"
		,Opportunity.Partnership_End_Date__c as "Partnership End Date"
		,User_.Name
		,Opportunity.Partner_Territory__c
		,Opportunity.Partnership_End_Date__c
FROM Account_
LEFT JOIN Opportunity
	ON Account_.Id_ = Opportunity.AccountId
LEFT JOIN RecordType
	ON Opportunity.RecordTypeId = RecordType.Id_
LEFT JOIN  User_
	ON Account_.Id_ = User_.AccountId
LEFT JOIN GroupMember
	ON User_.Id_ = GroupMember.UserOrGroupId
LEFT JOIN Group_
	ON GroupMember.UserOrGroupId = Group_.OwnerId
WHERE Account_.Active_Service_Partner__c = 'true'
	AND (RecordType.Name = 'Partnership')
	# AND (Opportunity.Partnership_Start_Date__c <= '2019-03-19') #today
	# AND (Opportunity.Partnership_End_Date__c like '%2018-12-30%')
	AND (Opportunity.Name not like '%deathray%')
	AND (Opportunity.Partner_Territory__c = '["United States"]');
